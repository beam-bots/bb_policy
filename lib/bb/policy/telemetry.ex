# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Telemetry do
  @moduledoc """
  Telemetry events emitted by `BB.Policy.Runner`.

  Events follow the Beam Bots convention `[:bb, <subsystem>, <operation>, <phase>]`
  and are emitted through `BB.Telemetry`, so they share the framework's
  observability tooling.

  ## Events

    * `[:bb, :policy, :episode, :start]` — measurements `%{system_time}`,
      metadata `%{robot, policy_module, goal}`.
    * `[:bb, :policy, :episode, :stop]` — measurements `%{steps}`,
      metadata `%{robot, policy_module, reason}`. `reason` is `:completed`,
      `:timeout`, `:disarmed`, or `{:error, term}`.
    * `[:bb, :policy, :inference, :stop]` — measurements `%{duration}` (native
      time units), metadata `%{robot, policy_module, step}`. Emitted per tick.
      Watch p99, not the mean — bounded worst-case latency is what matters for
      control.

  Attach with `:telemetry.attach/4` on these event names.
  """

  @doc false
  @spec episode_start(module(), module(), term()) :: :ok
  def episode_start(robot, policy_module, goal) do
    BB.Telemetry.emit(
      [:bb, :policy, :episode, :start],
      %{system_time: System.system_time()},
      %{robot: robot, policy_module: policy_module, goal: goal}
    )
  end

  @doc false
  @spec episode_stop(module(), module(), non_neg_integer(), term()) :: :ok
  def episode_stop(robot, policy_module, steps, reason) do
    BB.Telemetry.emit(
      [:bb, :policy, :episode, :stop],
      %{steps: steps},
      %{robot: robot, policy_module: policy_module, reason: reason}
    )
  end

  @doc false
  @spec inference_stop(module(), module(), non_neg_integer(), integer()) :: :ok
  def inference_stop(robot, policy_module, step, duration) do
    BB.Telemetry.emit(
      [:bb, :policy, :inference, :stop],
      %{duration: duration},
      %{robot: robot, policy_module: policy_module, step: step}
    )
  end
end
