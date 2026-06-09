# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ReactorIntegrationTest do
  @moduledoc """
  Proves `BB.Policy.Command` works as a real `bb_reactor` step — not just that
  it satisfies the command contract in isolation. A reactor with a `command`
  step pointing at the policy command runs against a live simulated robot.
  """
  use ExUnit.Case, async: false

  alias BB.Policy.Support.ReactorRobot
  alias BB.Reactor.Step.Command.Result

  defmodule PickReactor do
    @moduledoc false
    use Reactor, extensions: [BB.Reactor]

    command :pick do
      command(:pick)
    end

    return(:pick)
  end

  setup do
    start_supervised!({ReactorRobot, simulation: :kinematic})
    {:ok, cmd} = ReactorRobot.arm(%{})
    {:ok, :armed, _} = BB.Command.await(cmd, 5_000)
    :ok
  end

  test "a BB.Policy.Command runs as a reactor step and returns its Result" do
    assert {:ok, %Result{} = result} =
             Reactor.run(PickReactor, %{}, %{private: %{bb_robot: ReactorRobot}})

    assert result.command == :pick
    assert result.robot_module == ReactorRobot
    # The policy ran to completion; the reactor step unwraps the command result.
    assert result.outcome == :completed
  end
end
