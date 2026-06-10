# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Command do
  @moduledoc """
  Run a `BB.Policy` as a Beam Bots command.

  This is a `BB.Command` handler that executes a policy in the robot's command
  lifecycle and state machine. Declaring it on a robot makes a learned behaviour
  a first-class command — awaitable with `BB.Command.await/2`, governed by the
  safety system, and usable as a step in a `bb_reactor` workflow with no extra
  glue (the reactor's `command :name` step invokes it like any other command).

  ## Declaring a policy command

      defmodule MyRobot do
        use BB

        commands do
          command :pick_mug do
            handler {BB.Policy.Command,
              policy: BB.Policy.ONNX,
              policy_opts: [
                model: "priv/models/pick_mug.onnx",
                normalizer: "priv/models/pick_mug.json",
                observation: [positions: [:shoulder, :elbow, :wrist]],
                action: [{[:shoulder, :elbow, :wrist], :position}]
              ],
              rate_hz: 20}
            allowed_states [:idle]
            timeout :timer.seconds(30)
          end
        end
      end

      {:ok, cmd} = MyRobot.pick_mug(%{})
      {:ok, :completed} = BB.Command.await(cmd, 30_000)

  ## In a reactor workflow

      defmodule MyRobot.MakeCoffee do
        use Reactor, extensions: [BB.Reactor]

        command :pick_mug          # the learned policy command, above
        command :place_under_spout, do: command(:move_to)
        return :place_under_spout
      end

  ## Options

    * `:policy` (required) — a module implementing `BB.Policy`.
    * `:policy_opts` — keyword list passed to `c:BB.Policy.init/1`. Default `[]`.
    * `:rate_hz` — control-loop frequency. Default `20`.

  Episode duration is bounded by the command's `timeout` (declared in the DSL),
  not an option here — the command system owns the timeout timer.

  ## Lifecycle and outcomes

  Each tick runs one `BB.Policy.Step` while the robot is armed (the command
  system disarms-halts via `handle_safety_state_change/2`). The command stops
  with:

    * `{:ok, :completed}` — the policy returned `{:done, state}` from `act/2`.
    * `{:error, {:action_conversion, reason}}` — an action failed to convert.
    * exit `:timeout` — the command's timeout elapsed (→ `{:error, :timeout}`
      to awaiters).
    * exit `:disarmed` — the safety system disarmed mid-episode (a `bb_reactor`
      step surfaces this as `{:halt, :safety_disarmed}`).
  """

  use BB.Command,
    options_schema: [
      policy: [type: :atom, required: true, doc: "Module implementing BB.Policy"],
      policy_opts: [
        type: :keyword_list,
        default: [],
        doc: "Options passed to the policy's init/1"
      ],
      rate_hz: [type: :pos_integer, default: 20, doc: "Control-loop frequency (Hz)"]
    ]

  alias BB.Policy.Step
  alias BB.Policy.Telemetry

  @default_rate_hz 20

  @impl BB.Command
  def init(opts) do
    bb = Keyword.fetch!(opts, :bb)
    policy_module = Keyword.fetch!(opts, :policy)
    policy_opts = Keyword.get(opts, :policy_opts, [])
    rate_hz = Keyword.get(opts, :rate_hz, @default_rate_hz)

    case policy_module.init(policy_opts) do
      {:ok, policy_state} ->
        {:ok,
         %{
           robot: bb.robot,
           policy_module: policy_module,
           policy_state: policy_state,
           rate_hz: rate_hz,
           interval_ms: max(1, div(1000, rate_hz)),
           step: 0,
           result: nil
         }}

      {:error, reason} ->
        {:stop, {:policy_init, reason}}
    end
  end

  @impl BB.Command
  def handle_command(goal, context, state) do
    state = %{state | policy_state: state.policy_module.reset(state.policy_state)}
    Telemetry.episode_start(context.robot_module, state.policy_module, goal)
    schedule_tick(state)
    {:noreply, state}
  end

  @impl BB.Command
  def handle_info(:tick, state) do
    case Step.run(state.policy_module, state.policy_state, state.robot) do
      {:done, policy_state} ->
        state = %{state | policy_state: policy_state}
        episode_stop(state, :completed)
        {:stop, :normal, %{state | result: {:ok, :completed}}}

      {:applied, policy_state, %{inference_duration: duration}} ->
        Telemetry.inference_stop(state.robot, state.policy_module, state.step, duration)
        state = %{state | policy_state: policy_state, step: state.step + 1}
        schedule_tick(state)
        {:noreply, state}

      {:disarmed, policy_state} ->
        # Disarmed between the entry gate and apply; nothing was applied. End the
        # command as a safety stop (the core BB.Command safety path also halts on
        # disarm via handle_safety_state_change/2).
        state = %{state | policy_state: policy_state}
        episode_stop(state, :disarmed)
        {:stop, :normal, %{state | result: {:error, :disarmed}}}

      {:error, reason} ->
        episode_stop(state, reason)
        {:stop, :normal, %{state | result: {:error, reason}}}
    end
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl BB.Command
  def result(%{result: nil}), do: {:error, :incomplete}
  def result(%{result: result}), do: result

  defp schedule_tick(state), do: Process.send_after(self(), :tick, state.interval_ms)

  defp episode_stop(state, reason),
    do: Telemetry.episode_stop(state.robot, state.policy_module, state.step, reason)
end
