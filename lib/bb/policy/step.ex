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

  It is deliberately free of scheduling, timing, safety, and telemetry policy —
  callers (`BB.Policy.Runner`, `BB.Policy.Command`) own those. The caller is
  responsible for the safety gate: only call `run/3` when the robot is armed.

  `run/3` returns one of:

    * `{:done, policy_state}` — the policy signalled completion (`act/2`
      returned `{:done, state}`); no commands were applied.
    * `{:applied, policy_state, measurements}` — an action was applied;
      `measurements` carries `:inference_duration` (native time units).
    * `{:error, {:action_conversion, reason}}` — `action_to_commands/3` failed.
  """

  alias BB.Policy.Effect
  alias BB.Robot.Runtime

  @type outcome ::
          {:done, BB.Policy.state()}
          | {:applied, BB.Policy.state(), %{inference_duration: integer()}}
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
        Enum.each(commands, &Effect.apply(&1, robot))
        {:applied, policy_state, %{inference_duration: duration}}

      {:error, reason} ->
        {:error, {:action_conversion, reason}}
    end
  end
end
