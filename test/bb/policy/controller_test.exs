# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ControllerTest do
  use ExUnit.Case, async: false

  use Mimic

  alias BB.Policy.Controller
  alias BB.Policy.Support.MockPolicy

  @robot MyRobot

  setup :set_mimic_global

  setup do
    stub(BB.Robot.Runtime, :get_robot_state, fn @robot -> %{} end)
    stub(BB.Actuator, :set_position, fn @robot, _path, _value, _opts -> :ok end)
    :ok
  end

  defp init_opts(extra) do
    Keyword.merge([bb: %{robot: @robot}, policy: MockPolicy, policy_opts: [], rate: 50], extra)
  end

  describe "init/1" do
    test "initialises the policy and seeds loop state" do
      assert {:ok, state} = Controller.init(init_opts([]))
      assert state.robot == @robot
      assert state.rate == 50
      assert state.step == 0
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

      assert {:stop, {:policy_init, :no_model}} = Controller.init(init_opts(policy: BadPolicy))
    end
  end

  describe "tick loop" do
    test "runs a step and applies commands while armed" do
      parent = self()
      stub(BB.Safety, :armed?, fn @robot -> true end)

      stub(BB.Actuator, :set_position, fn @robot, path, _value, _ ->
        send(parent, {:applied, path})
        :ok
      end)

      {:ok, state} = Controller.init(init_opts(policy_opts: [path: [:hip]]))
      assert {:noreply, state} = Controller.handle_info(:tick, state)
      assert state.step == 1
      assert_received {:applied, [:hip]}
    end

    test "idles and does not command while disarmed" do
      parent = self()
      stub(BB.Safety, :armed?, fn @robot -> false end)

      stub(BB.Actuator, :set_position, fn @robot, _p, _v, _ ->
        send(parent, :should_not_happen)
        :ok
      end)

      {:ok, state} = Controller.init(init_opts([]))
      assert {:noreply, state} = Controller.handle_info(:tick, state)
      assert state.step == 0
      refute_received :should_not_happen
    end

    test "resets the policy once on the armed -> disarmed transition, not every idle tick" do
      parent = self()
      Mimic.copy(MockPolicy)
      stub(MockPolicy, :reset, fn s -> send(parent, :reset) && s end)

      # A toggleable armed flag (robust to armed? being consulted more than once
      # per tick — both the entry gate and Step's pre-apply re-check call it).
      {:ok, armed} = Agent.start_link(fn -> true end)
      stub(BB.Safety, :armed?, fn @robot -> Agent.get(armed, & &1) end)

      {:ok, state} = Controller.init(init_opts([]))
      # init/1 resets once; drain that.
      assert_received :reset

      # tick 1: armed -> runs a step, no reset
      {:noreply, state} = Controller.handle_info(:tick, state)
      refute_received :reset

      # disarm, then tick 2: armed -> disarmed transition -> exactly one reset
      Agent.update(armed, fn _ -> false end)
      {:noreply, state} = Controller.handle_info(:tick, state)
      assert_received :reset

      # tick 3: still disarmed -> no further reset (the fix)
      {:noreply, _state} = Controller.handle_info(:tick, state)
      refute_received :reset
    end

    test "a :done policy resets and keeps running (no terminal state)" do
      stub(BB.Safety, :armed?, fn @robot -> true end)

      # done_after: 0 -> MockPolicy returns :done once steps > 0, i.e. the first tick.
      {:ok, state} = Controller.init(init_opts(policy_opts: [done_after: 0]))
      assert {:noreply, state} = Controller.handle_info(:tick, state)
      # still alive, step not advanced (the step was a :done, which resets)
      assert state.step == 0
      assert {:noreply, _state} = Controller.handle_info(:tick, state)
    end
  end

  describe "disarm/1" do
    test "is a no-op (the policy emits no holding command of its own)" do
      assert Controller.disarm([]) == :ok
    end
  end
end
