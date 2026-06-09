# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Support.ReactorRobot do
  @moduledoc """
  A test robot whose `:pick` command is a `BB.Policy.Command` driving the
  dependency-free `MockPolicy`. Used to exercise `BB.Policy.Command` as a real
  `bb_reactor` step (no Ortex required). Run in `simulation: :kinematic`.
  """
  use BB

  alias BB.Policy.Support.MockPolicy

  commands do
    command :arm do
      handler(BB.Command.Arm)
      allowed_states([:disarmed])
    end

    command :disarm do
      handler(BB.Command.Disarm)
      allowed_states(:*)
    end

    # The policy-as-command under test: MockPolicy signals :done after 2 steps,
    # so the command completes promptly with {:ok, :completed}.
    command :pick do
      handler(
        {BB.Policy.Command,
         policy: MockPolicy, policy_opts: [done_after: 2, path: [:a, :a_servo]], rate_hz: 1000}
      )

      allowed_states([:idle])
    end
  end

  topology do
    link :base do
      joint :a do
        type(:revolute)

        limit do
          lower(~u(-90 degree))
          upper(~u(90 degree))
          effort(~u(5 newton_meter))
          velocity(~u(120 degree_per_second))
        end

        actuator(:a_servo, BB.Sim.Actuator)
        link(:tip)
      end
    end
  end
end
