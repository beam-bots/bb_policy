# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BbPolicyFirmware.Robot do
  @moduledoc """
  A minimal robot for the on-device end-to-end test: a base link with three
  revolute joints (so it matches the 3-input / 2-output linear test policy's
  joint set). Run in `simulation: :kinematic`, so `BB.Sim.Actuator` stands in
  for real servos — no hardware required to exercise the policy→actuator loop.
  """
  use BB

  commands do
    command :arm do
      handler(BB.Command.Arm)
      allowed_states([:disarmed])
    end

    command :disarm do
      handler(BB.Command.Disarm)
      allowed_states([:idle])
    end
  end

  topology do
    link :base_link do
      visual do
        cylinder do
          radius(~u(0.03 meter))
          height(~u(0.04 meter))
        end
      end

      joint :a do
        type(:revolute)

        origin do
          z(~u(0.04 meter))
        end

        limit do
          lower(~u(-90 degree))
          upper(~u(90 degree))
          effort(~u(5 newton_meter))
          velocity(~u(120 degree_per_second))
        end

        actuator(:a_servo, BB.Sim.Actuator)

        link :link_b do
          joint :b do
            type(:revolute)

            origin do
              z(~u(0.03 meter))
            end

            limit do
              lower(~u(-90 degree))
              upper(~u(90 degree))
              effort(~u(5 newton_meter))
              velocity(~u(120 degree_per_second))
            end

            actuator(:b_servo, BB.Sim.Actuator)

            link :link_c do
              joint :c do
                type(:revolute)

                origin do
                  z(~u(0.03 meter))
                end

                limit do
                  lower(~u(-90 degree))
                  upper(~u(90 degree))
                  effort(~u(5 newton_meter))
                  velocity(~u(120 degree_per_second))
                end

                actuator(:c_servo, BB.Sim.Actuator)

                link(:tip)
              end
            end
          end
        end
      end
    end
  end
end
