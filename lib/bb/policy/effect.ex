# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defprotocol BB.Policy.Effect do
  @moduledoc """
  An effect a policy emits for the runtime to apply on a control tick.

  `c:BB.Policy.action_to_commands/3` returns a list of effects; the control loop
  (`BB.Policy.Step`) applies each through `apply/2` while the robot is armed.
  Decoupling the *return type* from a concrete struct means the action
  vocabulary can grow without a breaking change to the `BB.Policy` contract: a
  new effect is a new struct implementing this protocol, not a new return shape.

  `BB.Policy.ActuatorCommand` is the built-in implementation (drive an actuator
  to a setpoint, or hold/stop it). Downstream code can add its own effects —
  publishing a message, issuing a sub-command, toggling an indicator — by
  defining a struct and `defimpl BB.Policy.Effect`.

  Implementations apply the effect for its side-effect and return `:ok`. The
  safety gate is the caller's responsibility: `BB.Policy.Step` only applies
  effects while `BB.Safety.armed?/1`.
  """

  @doc """
  Apply this effect to `robot`. Returns `:ok`.
  """
  @spec apply(t(), robot :: module()) :: :ok
  def apply(effect, robot)
end
