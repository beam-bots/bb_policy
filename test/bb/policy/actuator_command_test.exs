# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ActuatorCommandTest do
  use ExUnit.Case, async: false

  use Mimic

  alias BB.Policy.ActuatorCommand

  @robot MyRobot

  describe "builders" do
    test "position/3, velocity/3, effort/3 set kind and value" do
      assert %ActuatorCommand{kind: :position, value: 1.0, opts: [velocity: 2.0]} =
               ActuatorCommand.position(:j, 1.0, velocity: 2.0)

      assert %ActuatorCommand{kind: :velocity, value: 0.5, opts: []} =
               ActuatorCommand.velocity(:j, 0.5)

      assert %ActuatorCommand{kind: :effort, value: 3.0} = ActuatorCommand.effort(:j, 3.0)
    end
  end

  describe "apply/2" do
    test "dispatches each kind to the matching BB.Actuator call, wrapping a bare path" do
      parent = self()
      stub(BB.Actuator, :set_position, fn @robot, p, v, o -> send(parent, {:pos, p, v, o}) end)
      stub(BB.Actuator, :set_velocity, fn @robot, p, v, o -> send(parent, {:vel, p, v, o}) end)
      stub(BB.Actuator, :set_effort, fn @robot, p, v, o -> send(parent, {:eff, p, v, o}) end)

      ActuatorCommand.apply(ActuatorCommand.position(:servo, 1.0, velocity: 2.0), @robot)
      ActuatorCommand.apply(ActuatorCommand.velocity([:a, :b], 0.5), @robot)
      ActuatorCommand.apply(ActuatorCommand.effort([:c], 3.0), @robot)

      assert_received {:pos, [:servo], 1.0, [velocity: 2.0]}
      assert_received {:vel, [:a, :b], 0.5, []}
      assert_received {:eff, [:c], 3.0, []}
    end
  end
end
