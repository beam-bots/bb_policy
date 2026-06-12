# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Runner do
  @moduledoc """
  Executes a `BB.Policy` in a fixed-rate control loop.

  The runner is a `GenServer` that, on each tick, drives the policy cycle and
  applies the result to the robot's actuators — subject to the safety system:

      observe/3  →  act/2  →  action_to_commands/3  →  (safety check)  →  apply

  It owns:

    * **Observation collection** — reads robot state from `BB.Robot.Runtime` and
      hands it to the policy's `c:BB.Policy.observe/3`.
    * **Inference scheduling** — ticks at `:rate_hz`, rescheduled per tick via
      `Process.send_after/3` (the ecosystem idiom; see `BB.PID.Controller`).
    * **Action application** — applies `BB.Policy.ActuatorCommand`s via
      `BB.Actuator`, but only while `BB.Safety.armed?/1` is true. A disarm halts
      the episode.
    * **Episode lifecycle** — calls `c:BB.Policy.reset/1` at episode start and
      ends on completion (`{:done, state}` from `act/2`), timeout, safety
      intervention, or cancellation.
    * **Telemetry** — emits `[:bb, :policy, …]` events (see `BB.Policy.Telemetry`).

  ## Entry point

  Use `run/4` for an episodic task that runs to completion or timeout:

      {:ok, result} =
        BB.Policy.Runner.run(MyRobot, BB.Policy.ONNX, %{task: :pick_mug},
          policy_opts: [
            model: "priv/models/pick_mug.onnx",
            normalizer: "priv/models/pick_mug.json",
            observation_keys: [:joint_positions, :joint_velocities],
            action_keys: [:target_positions]
          ],
          rate_hz: 20,
          timeout: :timer.seconds(30)
        )

  `run/4` blocks until the episode ends and returns `{:ok, reason}` where
  `reason` is `:completed`, `:timeout`, or `:disarmed`; or `{:error, term}` if
  the policy fails to initialise or an action conversion errors.

  > #### Public API note {: .info}
  >
  > The proposal also describes a `run_policy/4` convenience function on `BB.Motion`.
  > `BB.Motion` lives in the `bb` core package and has no extension hook for
  > satellites, so that
  > convenience delegate must land via a PR to core. Until then, call
  > `BB.Policy.Runner.run/4` (or the `BB.Policy.run/4` facade) directly.

  ## Options

    * `:rate_hz` — control-loop frequency. Default `20`.
    * `:timeout` — maximum episode duration in ms. Default `30_000`.
    * `:policy_opts` — keyword list passed to `c:BB.Policy.init/1`.
    * `:goal` — goal specification forwarded to the policy.
  """

  use GenServer

  alias BB.Policy.Step
  alias BB.Policy.Telemetry
  alias BB.Process, as: BBProcess

  @default_rate_hz 20
  @default_timeout :timer.seconds(30)

  @type reason :: :completed | :timeout | :disarmed | {:error, term()}
  @type run_result :: {:ok, reason()} | {:error, term()}

  defstruct [
    :robot,
    :policy_module,
    :policy_state,
    :goal,
    :rate_hz,
    :interval_ms,
    :timeout,
    :deadline,
    :tick_ref,
    :owner,
    episode_step: 0
  ]

  @typedoc false
  @type t :: %__MODULE__{}

  @doc """
  Start a runner under the robot's registry as a named process.

  Prefer `run/4` for one-shot episodic execution; use `start_link/1` when you
  want to supervise a long-lived policy process yourself.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    robot = Keyword.fetch!(opts, :robot)
    GenServer.start_link(__MODULE__, opts, name: BBProcess.via(robot, :policy_runner))
  end

  @doc """
  Run a policy on `robot` until completion, timeout, or safety intervention.

  Blocks the caller until the episode finishes and returns its result. See the
  moduledoc for options and return values.
  """
  @spec run(robot :: module(), policy :: module(), term(), keyword()) :: run_result()
  def run(robot, policy_module, goal, opts \\ []) do
    opts =
      opts
      |> Keyword.merge(robot: robot, policy: policy_module, goal: goal, owner: self())

    # Unnamed, transient runner: an episode is a one-shot, and run/4 may be
    # called repeatedly, so we don't register it under the robot's :policy_runner.
    case GenServer.start(__MODULE__, opts) do
      {:ok, pid} -> await_episode(pid)
      {:error, reason} -> {:error, reason}
    end
  end

  defp await_episode(pid) do
    ref = Process.monitor(pid)

    receive do
      {:episode_result, ^pid, result} ->
        Process.demonitor(ref, [:flush])
        result

      {:DOWN, ^ref, :process, ^pid, :normal} ->
        # Stopped without reporting (shouldn't happen) — treat as completed.
        {:ok, :completed}

      {:DOWN, ^ref, :process, ^pid, down_reason} ->
        {:error, down_reason}
    end
  end

  @impl GenServer
  def init(opts) do
    robot = Keyword.fetch!(opts, :robot)
    policy_module = Keyword.fetch!(opts, :policy)
    policy_opts = Keyword.get(opts, :policy_opts, [])
    rate_hz = Keyword.get(opts, :rate_hz, @default_rate_hz)
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    goal = Keyword.get(opts, :goal)

    case policy_module.init(policy_opts) do
      {:ok, policy_state} ->
        state = %__MODULE__{
          robot: robot,
          policy_module: policy_module,
          policy_state: policy_module.reset(policy_state),
          goal: goal,
          rate_hz: rate_hz,
          interval_ms: max(1, div(1000, rate_hz)),
          timeout: timeout,
          deadline: monotonic_ms() + timeout,
          owner: Keyword.get(opts, :owner),
          episode_step: 0
        }

        Telemetry.episode_start(robot, policy_module, goal)
        {:ok, schedule_tick(state), {:continue, :first_tick}}

      {:error, reason} ->
        {:stop, {:policy_init, reason}}
    end
  end

  @impl GenServer
  def handle_continue(:first_tick, state), do: {:noreply, state}

  @impl GenServer
  def handle_info(:tick, %__MODULE__{} = state) do
    cond do
      monotonic_ms() >= state.deadline ->
        finish(state, :timeout)

      not BB.Safety.armed?(state.robot) ->
        # A disarm (or never having armed) is a safety intervention, not a
        # retryable error — end the episode.
        finish(state, :disarmed)

      true ->
        run_step(state)
    end
  end

  @impl GenServer
  def terminate(_reason, _state), do: :ok

  # --- the control step ----------------------------------------------------

  defp run_step(%__MODULE__{} = state) do
    case Step.run(state.policy_module, state.policy_state, state.robot) do
      {:done, ps} ->
        finish(%{state | policy_state: ps}, :completed)

      {:applied, ps, %{inference_duration: duration}} ->
        Telemetry.inference_stop(state.robot, state.policy_module, state.episode_step, duration)

        {:noreply,
         schedule_tick(%{state | policy_state: ps, episode_step: state.episode_step + 1})}

      {:disarmed, ps} ->
        # Disarmed mid-tick (between the entry gate and apply); nothing was
        # applied. Same intervention as a top-of-tick disarm — end the episode.
        finish(%{state | policy_state: ps}, :disarmed)

      {:error, _reason} = error ->
        finish(state, error)
    end
  end

  # --- episode termination -------------------------------------------------

  defp finish(%__MODULE__{} = state, reason) do
    Telemetry.episode_stop(state.robot, state.policy_module, state.episode_step, reason)
    report(state, reason)
    {:stop, :normal, state}
  end

  defp report(%__MODULE__{owner: nil}, _reason), do: :ok

  defp report(%__MODULE__{owner: owner} = state, reason) do
    send(owner, {:episode_result, self(), result_for(reason)})
    _ = state
    :ok
  end

  defp result_for({:error, _} = error), do: error
  defp result_for(reason), do: {:ok, reason}

  # --- helpers -------------------------------------------------------------

  defp schedule_tick(%__MODULE__{interval_ms: interval_ms} = state) do
    %{state | tick_ref: Process.send_after(self(), :tick, interval_ms)}
  end

  defp monotonic_ms, do: System.monotonic_time(:millisecond)
end
