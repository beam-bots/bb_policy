# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Support.MockPolicy do
  @moduledoc """
  A trivial, dependency-free `BB.Policy` for tests.

  It carries a step counter, "observes" nothing, and "acts" by emitting a
  single position command on a configurable actuator path. Use it to exercise
  `BB.Policy.Runner` lifecycle and the behaviour contract without Ortex or a
  real model.

  ## Options (`policy_opts`)

    * `:done_after` — end the episode (`act/2` returns `:done`) once this many
      steps have run. Omit to run until timeout/disarm.
    * `:path` — actuator path for the emitted command. Default `[:joint, :servo]`.
    * `:fail_commands` — when `true`, `action_to_commands/3` returns an error,
      to test the runner's error path.
  """

  @behaviour BB.Policy

  alias BB.Policy.ActuatorCommand

  @impl BB.Policy
  def init(opts) do
    {:ok,
     %{
       steps: 0,
       done_after: Keyword.get(opts, :done_after),
       path: Keyword.get(opts, :path, [:joint, :servo]),
       fail_commands: Keyword.get(opts, :fail_commands, false)
     }}
  end

  @impl BB.Policy
  def reset(state), do: %{state | steps: 0}

  @impl BB.Policy
  def observe(_robot_state, _sensors, state) do
    {%{noop: Nx.tensor([0.0])}, %{state | steps: state.steps + 1}}
  end

  @impl BB.Policy
  def act(_observation, %{done_after: n, steps: steps} = state)
      when is_integer(n) and steps > n do
    {:done, state}
  end

  def act(_observation, state) do
    {%{target_positions: Nx.tensor([0.0])}, state}
  end

  @impl BB.Policy
  def action_to_commands(_action, _robot, %{fail_commands: true}), do: {:error, :boom}

  def action_to_commands(_action, _robot, state) do
    {:ok, [ActuatorCommand.position(state.path, 0.0)]}
  end

  @impl BB.Policy
  def info(state), do: %{architecture: :mock, steps: state.steps}
end
