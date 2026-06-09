# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Controller do
  @moduledoc """
  Run a `BB.Policy` as a long-lived `BB.Controller`.

  Where `BB.Policy.Runner` and `BB.Policy.Command` run a policy as a *bounded
  episode* (it ends on completion, timeout, or cancellation), this controller
  runs a policy *continuously* as part of the robot's supervision tree — the
  idiom for a reactive/standing behaviour. It is declared in the robot DSL like
  any other controller (compare `BB.PID.Controller`), gets supervised and
  parameter-aware for free, and exposes a safety `disarm/1`.

  Each tick, while the robot is armed, it runs one `BB.Policy.Step`. When the
  robot is not armed it idles (and resets the policy so the next armed run
  starts from a clean episode). A policy that returns `{:done, state}` from
  `act/2` simply resets and keeps running — "done" has no terminal meaning for a
  standing controller; use `BB.Policy.Command` when you want an episode that
  ends.

  ## Declaring a policy controller

      defmodule MyRobot do
        use BB

        controllers do
          controller :balance,
            {BB.Policy.Controller,
             policy: BB.Policy.ONNX,
             policy_opts: [
               model: "priv/models/balance.onnx",
               observation: [positions: [:hip, :knee], velocities: [:hip, :knee]],
               action: [{[:hip, :knee], :effort}]
             ],
             rate: 50}
        end
      end

  ## Options

    * `:policy` (required) — a module implementing `BB.Policy`.
    * `:policy_opts` — keyword list passed to `c:BB.Policy.init/1`. Default `[]`.
    * `:rate` — control-loop frequency in Hz. Default `20`.

  ## Telemetry

  Emits per-tick `[:bb, :policy, :inference, :stop]` events. It does not emit
  episode start/stop events — a standing controller has no episode boundary.
  """

  use BB.Controller,
    options_schema: [
      policy: [type: :atom, required: true, doc: "Module implementing BB.Policy"],
      policy_opts: [
        type: :keyword_list,
        default: [],
        doc: "Options passed to the policy's init/1"
      ],
      rate: [type: :pos_integer, default: 20, doc: "Control-loop frequency (Hz)"]
    ]

  alias BB.Policy.Step
  alias BB.Policy.Telemetry

  @impl BB.Controller
  def init(opts) do
    bb = Keyword.fetch!(opts, :bb)
    policy_module = Keyword.fetch!(opts, :policy)
    policy_opts = Keyword.fetch!(opts, :policy_opts)
    rate = Keyword.fetch!(opts, :rate)

    case policy_module.init(policy_opts) do
      {:ok, policy_state} ->
        state = %{
          robot: bb.robot,
          policy_module: policy_module,
          policy_state: policy_module.reset(policy_state),
          rate: rate,
          step: 0
        }

        schedule_tick(rate)
        {:ok, state}

      {:error, reason} ->
        {:stop, {:policy_init, reason}}
    end
  end

  @impl BB.Controller
  def handle_info(:tick, state) do
    state =
      if BB.Safety.armed?(state.robot) do
        run_step(state)
      else
        # Idle while disarmed; keep the policy ready for a fresh armed episode.
        %{state | policy_state: state.policy_module.reset(state.policy_state)}
      end

    schedule_tick(state.rate)
    {:noreply, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl BB.Controller
  def disarm(_opts), do: :ok

  defp run_step(state) do
    case Step.run(state.policy_module, state.policy_state, state.robot) do
      {:done, policy_state} ->
        # A standing controller has no terminal state — reset and continue.
        %{state | policy_state: state.policy_module.reset(policy_state)}

      {:applied, policy_state, %{inference_duration: duration}} ->
        Telemetry.inference_stop(state.robot, state.policy_module, state.step, duration)
        %{state | policy_state: policy_state, step: state.step + 1}

      {:error, _reason} ->
        # Degrade gracefully: skip this tick, keep running. The next tick retries.
        state
    end
  end

  defp schedule_tick(rate), do: Process.send_after(self(), :tick, max(1, div(1000, rate)))
end
