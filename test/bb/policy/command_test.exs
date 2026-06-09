# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.CommandTest do
  use ExUnit.Case, async: false

  use Mimic

  alias BB.Policy.Command
  alias BB.Policy.Support.MockPolicy

  @robot MyRobot

  setup :set_mimic_global

  setup do
    stub(BB.Robot.Runtime, :get_robot_state, fn @robot -> %{} end)
    stub(BB.Actuator, :set_position, fn @robot, _path, _value, _opts -> :ok end)
    :ok
  end

  # The command server merges `bb: %{robot: ...}`, `goal:`, `context:` into the
  # handler opts before calling init/1. We replicate that envelope here.
  defp init_opts(extra) do
    Keyword.merge([bb: %{robot: @robot}, policy: MockPolicy, policy_opts: []], extra)
  end

  defp context, do: %{robot_module: @robot}

  describe "init/1" do
    test "initialises the policy and seeds loop state" do
      assert {:ok, state} = Command.init(init_opts(rate_hz: 50))
      assert state.robot == @robot
      assert state.policy_module == MockPolicy
      assert state.rate_hz == 50
      assert state.interval_ms == 20
      assert state.result == nil
    end

    test "stops when the policy fails to initialise" do
      defmodule BadPolicy do
        @behaviour BB.Policy
        def init(_), do: {:error, :no_model}
        def reset(s), do: s
        def observe(_, _, s), do: {%{}, s}
        def act(_, s), do: {%{}, s}
        def action_to_commands(_, _, _), do: {:ok, []}
      end

      assert {:stop, {:policy_init, :no_model}} =
               Command.init(init_opts(policy: BadPolicy))
    end
  end

  describe "handle_command/3 + tick loop" do
    test "schedules a tick and runs until the policy signals :done" do
      {:ok, state} = Command.init(init_opts(policy_opts: [done_after: 2], rate_hz: 1000))
      assert {:noreply, state} = Command.handle_command(%{task: :noop}, context(), state)

      # Drive ticks manually (the server would deliver these from send_after).
      # MockPolicy increments steps in observe/3 and returns :done once steps > 2,
      # so the third tick completes the episode.
      assert {:noreply, state} = Command.handle_info(:tick, state)
      assert {:noreply, state} = Command.handle_info(:tick, state)
      assert {:stop, :normal, final} = Command.handle_info(:tick, state)
      assert final.result == {:ok, :completed}
    end

    test "applies actuator commands on each applied tick" do
      parent = self()

      stub(BB.Actuator, :set_position, fn @robot, path, value, _ ->
        send(parent, {:applied, path, value})
        :ok
      end)

      {:ok, state} = Command.init(init_opts(policy_opts: [path: [:wrist, :servo]], rate_hz: 1000))
      {:noreply, state} = Command.handle_command(%{}, context(), state)
      {:noreply, _state} = Command.handle_info(:tick, state)

      assert_received {:applied, [:wrist, :servo], +0.0}
    end

    test "stops with an error when action conversion fails" do
      {:ok, state} = Command.init(init_opts(policy_opts: [fail_commands: true], rate_hz: 1000))
      {:noreply, state} = Command.handle_command(%{}, context(), state)

      assert {:stop, :normal, final} = Command.handle_info(:tick, state)
      assert final.result == {:error, {:action_conversion, :boom}}
    end
  end

  describe "result/1" do
    test "returns the stored result" do
      assert Command.result(%{result: {:ok, :completed}}) == {:ok, :completed}
      assert Command.result(%{result: {:error, :x}}) == {:error, :x}
    end

    test "reports :incomplete if the command never finished" do
      assert Command.result(%{result: nil}) == {:error, :incomplete}
    end
  end

  describe "safety integration" do
    test "the default disarm handler stops the command with :disarmed" do
      {:ok, state} = Command.init(init_opts([]))
      # handle_safety_state_change/2 comes from `use BB.Command` (default).
      assert {:stop, :disarmed, ^state} = Command.handle_safety_state_change(:disarmed, state)
    end
  end
end
