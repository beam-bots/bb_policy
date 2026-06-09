# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BbPolicyFirmware.Application do
  @moduledoc false
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = children(target())

    opts = [strategy: :one_for_one, name: BbPolicyFirmware.Supervisor]
    {:ok, sup} = Supervisor.start_link(children, opts)

    Logger.info("bb_policy_firmware started on target #{inspect(target())}")
    Logger.info("Run BbPolicyFirmware.Bench.run() to exercise the policy end-to-end")
    {:ok, sup}
  end

  # On the host and on the device we run the robot in kinematic simulation:
  # BB.Sim.Actuator stands in for real servos, so the policy→actuator loop runs
  # without hardware. (Swap to real actuators + hardware config when wired up.)
  defp children(_target) do
    [
      {BbPolicyFirmware.Robot, simulation: :kinematic}
    ]
  end

  def target do
    Application.get_env(:bb_policy_firmware, :target, :host)
  end
end
