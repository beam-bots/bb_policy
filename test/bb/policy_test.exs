# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.PolicyTest do
  use ExUnit.Case, async: true

  alias BB.Policy.Support.MockPolicy

  describe "behaviour contract" do
    test "BB.Policy defines the expected callbacks" do
      callbacks = BB.Policy.behaviour_info(:callbacks)

      assert {:init, 1} in callbacks
      assert {:reset, 1} in callbacks
      assert {:observe, 3} in callbacks
      assert {:act, 2} in callbacks
      assert {:action_to_commands, 3} in callbacks
    end

    test "info/1 is optional" do
      assert {:info, 1} in BB.Policy.behaviour_info(:optional_callbacks)
    end
  end

  describe "a conforming implementation (MockPolicy)" do
    test "init/1 returns initial state" do
      assert {:ok, state} = MockPolicy.init(model: "noop")
      assert state.steps == 0
    end

    test "the observe -> act -> action_to_commands cycle threads state" do
      assert {:ok, state} = MockPolicy.init([])
      assert {_obs, state} = MockPolicy.observe(%{}, %{}, state)
      assert state.steps == 1
      assert {action, state} = MockPolicy.act(%{}, state)
      assert %{target_positions: _} = action
      assert {:ok, commands} = MockPolicy.action_to_commands(action, :robot, state)
      assert is_list(commands)
    end

    test "reset/1 clears accumulated state" do
      {:ok, state} = MockPolicy.init([])
      {_obs, state} = MockPolicy.observe(%{}, %{}, state)
      assert state.steps == 1
      assert MockPolicy.reset(state).steps == 0
    end
  end
end
