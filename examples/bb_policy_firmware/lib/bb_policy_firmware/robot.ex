# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BbPolicyFirmware.Robot do
  @moduledoc """
  A minimal robot for the on-device end-to-end test: a base link with three
  revolute joints (so it matches the 3-input / 2-output linear test policy's
  joint set). Run in `simulation: :kinematic`, so `BB.Sim.Actuator` stands in
  for real servos — no hardware required to exercise the policy→actuator loop.

  It also declares a `BB.Policy.Controller` so the *DSL* path (a supervised,
  standing policy controller) is exercised end-to-end, not just the imperative
  `BB.Policy.run/4` path in `BbPolicyFirmware.Bench`. The controller is declared
  `simulation: :start` so it runs under `:kinematic` sim (controllers default to
  `:omit`, which would not start in sim).
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

  controllers do
    controller(
      :policy,
      {BB.Policy.Controller,
       policy: BB.Policy.ONNX,
       policy_opts: [
         # ONNX passes :model straight to Ortex.load, which resolves it relative
         # to the working directory — so on a Nerves device this wants the
         # app_dir path (BbPolicyFirmware.Bench resolves the same file via
         # Application.app_dir/2). Shown relative here for readability.
         model: "priv/models/linear_policy.onnx",
         observation: [positions: [:a, :b, :c]],
         action: [{[:a, :b, :c], :position}]
       ],
       rate: 20},
      # Controllers default to simulation: :omit; :start runs the policy under
      # :kinematic sim too (this robot runs simulation: :kinematic).
      simulation: :start
    )
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
