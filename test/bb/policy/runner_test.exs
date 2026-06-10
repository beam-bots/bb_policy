# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.RunnerTest do
  use ExUnit.Case, async: false

  use Mimic

  alias BB.Policy.Runner
  alias BB.Policy.Support.MockPolicy

  @robot MyRobot

  @doc false
  def handle_telemetry(event, measurements, metadata, parent) do
    send(parent, {:telemetry, event, measurements, metadata})
  end

  # The runner runs in its own process (started by run/4), so stubs must apply
  # across processes — global mode (requires async: false).
  setup :set_mimic_global
  setup :verify_on_exit!

  setup do
    # Default happy-path stubs: robot is armed, has empty state, accepts commands.
    stub(BB.Safety, :armed?, fn @robot -> true end)
    stub(BB.Robot.Runtime, :get_robot_state, fn @robot -> %{} end)
    stub(BB.Actuator, :set_position, fn @robot, _path, _value, _opts -> :ok end)
    :ok
  end

  describe "run/4 lifecycle" do
    test "runs to completion when the policy signals :done" do
      assert {:ok, :completed} =
               Runner.run(@robot, MockPolicy, %{task: :noop},
                 policy_opts: [done_after: 3],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )
    end

    test "ends with :timeout when the policy never completes" do
      assert {:ok, :timeout} =
               Runner.run(@robot, MockPolicy, %{task: :forever},
                 policy_opts: [],
                 rate_hz: 200,
                 timeout: 50
               )
    end

    test "ends with :disarmed when the robot is not armed" do
      stub(BB.Safety, :armed?, fn @robot -> false end)

      assert {:ok, :disarmed} =
               Runner.run(@robot, MockPolicy, %{},
                 policy_opts: [],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )
    end

    test "ends with :disarmed when the robot disarms mid-episode" do
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      stub(BB.Safety, :armed?, fn @robot ->
        Agent.get_and_update(counter, fn n -> {n, n + 1} end) < 2
      end)

      assert {:ok, :disarmed} =
               Runner.run(@robot, MockPolicy, %{},
                 policy_opts: [],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )
    end

    test "returns the policy init error" do
      defmodule FailingInit do
        @behaviour BB.Policy
        def init(_), do: {:error, :bad_model}
        def reset(s), do: s
        def observe(_, _, s), do: {%{}, s}
        def act(_, s), do: {%{}, s}
        def action_to_commands(_, _, _s), do: {:ok, []}
      end

      assert {:error, {:policy_init, :bad_model}} =
               Runner.run(@robot, FailingInit, %{}, rate_hz: 200, timeout: 100)
    end

    test "propagates an action-conversion error" do
      assert {:error, {:action_conversion, :boom}} =
               Runner.run(@robot, MockPolicy, %{},
                 policy_opts: [fail_commands: true],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )
    end
  end

  describe "command application" do
    test "applies the policy's actuator commands while armed" do
      test_pid = self()

      stub(BB.Actuator, :set_position, fn @robot, path, value, _opts ->
        send(test_pid, {:applied, path, value})
        :ok
      end)

      assert {:ok, :completed} =
               Runner.run(@robot, MockPolicy, %{},
                 policy_opts: [done_after: 1, path: [:shoulder, :servo]],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )

      assert_received {:applied, [:shoulder, :servo], +0.0}
    end

    test "does not apply commands when disarmed" do
      test_pid = self()
      stub(BB.Safety, :armed?, fn @robot -> false end)

      stub(BB.Actuator, :set_position, fn @robot, _path, _value, _opts ->
        send(test_pid, :should_not_happen)
        :ok
      end)

      assert {:ok, :disarmed} =
               Runner.run(@robot, MockPolicy, %{}, policy_opts: [], rate_hz: 200, timeout: 200)

      refute_received :should_not_happen
    end

    test "a disarm during the tick (after the entry gate) is not applied" do
      # armed? is true at the top-of-tick entry gate but false at the re-check
      # Step.run does immediately before applying — the disarm landed during
      # observe/act. No actuator command must reach the robot.
      test_pid = self()
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      stub(BB.Safety, :armed?, fn @robot ->
        # call 0 = entry gate (armed), call 1 = Step re-check (disarmed)
        Agent.get_and_update(counter, fn n -> {n, n + 1} end) < 1
      end)

      stub(BB.Actuator, :set_position, fn @robot, _path, _value, _opts ->
        send(test_pid, :should_not_happen)
        :ok
      end)

      assert {:ok, :disarmed} =
               Runner.run(@robot, MockPolicy, %{},
                 policy_opts: [path: [:shoulder, :servo]],
                 rate_hz: 200,
                 timeout: 200
               )

      refute_received :should_not_happen
    end
  end

  describe "telemetry" do
    test "emits episode start/stop events" do
      ref = make_ref()
      parent = self()

      handler_id = "test-#{inspect(ref)}"

      :telemetry.attach_many(
        handler_id,
        [
          [:bb, :policy, :episode, :start],
          [:bb, :policy, :episode, :stop],
          [:bb, :policy, :inference, :stop]
        ],
        &__MODULE__.handle_telemetry/4,
        parent
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)

      assert {:ok, :completed} =
               Runner.run(@robot, MockPolicy, %{task: :pick},
                 policy_opts: [done_after: 2],
                 rate_hz: 200,
                 timeout: :timer.seconds(5)
               )

      assert_received {:telemetry, [:bb, :policy, :episode, :start], %{system_time: _},
                       %{policy_module: MockPolicy}}

      assert_received {:telemetry, [:bb, :policy, :inference, :stop], %{duration: _}, _}

      assert_received {:telemetry, [:bb, :policy, :episode, :stop], %{steps: steps},
                       %{reason: :completed}}

      assert steps >= 2
    end
  end
end
