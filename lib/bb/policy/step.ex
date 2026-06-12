# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Step do
  @moduledoc """
  One iteration of the policy control cycle, shared by the runner and the
  command wrapper.

  A *step* reads robot state, runs the policy, and applies the resulting
  commands to the actuators:

      observe/3  →  act/2  →  action_to_commands/3  →  apply

  It is deliberately free of scheduling, timing, and telemetry policy — callers
  (`BB.Policy.Runner`, `BB.Policy.Command`) own those. Callers still gate *entry*
  on `BB.Safety.armed?/1` (don't spend inference while disarmed, and decide
  episode termination), but `run/3` re-checks `armed?` once more immediately
  before applying any effect: a disarm that lands *during* inference must not
  result in actuator commands. If disarmed at that point, no effect is applied
  and `run/3` returns `{:disarmed, policy_state}`.

  `run/3` returns one of:

    * `{:done, policy_state}` — the policy signalled completion (`act/2`
      returned `{:done, state}`); no commands were applied.
    * `{:applied, policy_state, measurements}` — an action was applied;
      `measurements` carries `:inference_duration` (native time units).
    * `{:disarmed, policy_state}` — the robot was disarmed between entry and
      apply (a mid-inference safety intervention); no effect was applied.
    * `{:error, {:action_conversion, reason}}` — `action_to_commands/3` failed.
  """

  alias BB.Policy.Effect
  alias BB.Robot.Runtime

  @type outcome ::
          {:done, BB.Policy.state()}
          | {:applied, BB.Policy.state(), %{inference_duration: integer()}}
          | {:disarmed, BB.Policy.state()}
          | {:error, {:action_conversion, term()}}

  @doc """
  Run one control step for `policy_module` against `robot`.

  `sensors` defaults to an empty map until a sensor-collection pipeline lands.
  """
  @spec run(module(), BB.Policy.state(), module(), %{atom() => term()}) :: outcome()
  def run(policy_module, policy_state, robot, sensors \\ %{}) do
    robot_state = Runtime.get_robot_state(robot)

    started = System.monotonic_time()
    {observation, policy_state} = policy_module.observe(robot_state, sensors, policy_state)

    case policy_module.act(observation, policy_state) do
      {:done, policy_state} ->
        {:done, policy_state}

      {action, policy_state} ->
        duration = System.monotonic_time() - started
        apply_action(policy_module, policy_state, robot, action, duration)
    end
  end

  defp apply_action(policy_module, policy_state, robot, action, duration) do
    case policy_module.action_to_commands(action, robot, policy_state) do
      {:ok, commands} ->
        # Re-check the safety gate immediately before touching actuators: a
        # disarm during observe/act (inference can dominate the tick) must not
        # apply this tick's effects. The caller already gated entry; this closes
        # the window between that check and the apply.
        if BB.Safety.armed?(robot) do
          Enum.each(commands, &Effect.apply(&1, robot))
          {:applied, policy_state, %{inference_duration: duration}}
        else
          {:disarmed, policy_state}
        end

      {:error, reason} ->
        {:error, {:action_conversion, reason}}
    end
  end
end
