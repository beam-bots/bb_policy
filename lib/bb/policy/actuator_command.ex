# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ActuatorCommand do
  @moduledoc """
  A single actuator command produced by a policy.

  `c:BB.Policy.action_to_commands/3` returns a list of these; `BB.Policy.Runner`
  applies each one through `BB.Actuator` on the tick that produced it. The
  struct is a thin, declarative description of "drive this actuator to this
  value" — it carries no behaviour, so policies stay decoupled from the actuator
  transport and the runner owns dispatch (and the safety gate around it).

  ## Fields

    * `:path` — the actuator's path in the robot topology, e.g.
      `[:shoulder_pan, :shoulder_pan_servo]`. A bare atom is accepted and
      wrapped as a single-element path.
    * `:kind` — `:position`, `:velocity`, or `:effort`.
    * `:value` — the numeric setpoint, in SI units (radians, rad/s, N·m).
    * `:opts` — extra options forwarded to the `BB.Actuator` call (e.g.
      `velocity:` for a position command). Defaults to `[]`.

  ## Example

      %BB.Policy.ActuatorCommand{
        path: [:shoulder_pan, :shoulder_pan_servo],
        kind: :position,
        value: 0.5,
        opts: [velocity: 1.0]
      }
  """

  @enforce_keys [:path, :kind, :value]
  defstruct [:path, :kind, :value, opts: []]

  @type kind :: :position | :velocity | :effort

  @type t :: %__MODULE__{
          path: atom() | [atom()],
          kind: kind(),
          value: number(),
          opts: keyword()
        }

  @doc """
  Build a position command for `path`.

  `opts` are forwarded to `BB.Actuator.set_position/4` (e.g. `velocity:`).
  """
  @spec position(atom() | [atom()], number(), keyword()) :: t()
  def position(path, value, opts \\ []),
    do: %__MODULE__{path: path, kind: :position, value: value, opts: opts}

  @doc "Build a velocity command for `path`."
  @spec velocity(atom() | [atom()], number(), keyword()) :: t()
  def velocity(path, value, opts \\ []),
    do: %__MODULE__{path: path, kind: :velocity, value: value, opts: opts}

  @doc "Build an effort command for `path`."
  @spec effort(atom() | [atom()], number(), keyword()) :: t()
  def effort(path, value, opts \\ []),
    do: %__MODULE__{path: path, kind: :effort, value: value, opts: opts}

  @doc """
  Apply this command to `robot` via the corresponding `BB.Actuator` call.

  Returns `:ok`. Commands are delivered over PubSub (`BB.Actuator.set_*/4`) so
  they are logged and replayable; the safety gate is the caller's
  responsibility (`BB.Policy.Runner` only calls this while armed).
  """
  @spec apply(t(), robot :: module()) :: :ok
  def apply(%__MODULE__{} = command, robot) do
    path = List.wrap(command.path)

    case command.kind do
      :position -> BB.Actuator.set_position(robot, path, command.value, command.opts)
      :velocity -> BB.Actuator.set_velocity(robot, path, command.value, command.opts)
      :effort -> BB.Actuator.set_effort(robot, path, command.value, command.opts)
    end
  end
end
