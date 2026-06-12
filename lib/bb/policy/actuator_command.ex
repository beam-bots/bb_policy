# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ActuatorCommand do
  @moduledoc """
  A single actuator command produced by a policy — the built-in
  `BB.Policy.Effect`.

  `c:BB.Policy.action_to_commands/3` returns a list of effects; `BB.Policy.Step`
  applies each through `BB.Policy.Effect.apply/2` on the tick that produced it.
  This struct is a thin, declarative description of "drive this actuator" — it
  carries no behaviour, so policies stay decoupled from the actuator transport
  and the runner owns dispatch (and the safety gate around it).

  ## Fields

    * `:path` — the actuator's path in the robot topology, e.g.
      `[:shoulder_pan, :shoulder_pan_servo]`. A bare atom is accepted and
      wrapped as a single-element path.
    * `:kind` — the verb:
      * `:position` / `:velocity` / `:effort` — drive to a numeric setpoint
        (`BB.Actuator.set_position/velocity/effort`).
      * `:hold` — hold the current setpoint (`BB.Actuator.hold/3`).
      * `:stop` — stop the actuator (`BB.Actuator.stop/3`).
    * `:value` — the numeric setpoint, in SI units (radians, rad/s, N·m). Unused
      (and may be `nil`) for `:hold`/`:stop`.
    * `:opts` — extra options forwarded to the `BB.Actuator` call (e.g.
      `velocity:` for a position command). Defaults to `[]`.

  ## Example

      %BB.Policy.ActuatorCommand{
        path: [:shoulder_pan, :shoulder_pan_servo],
        kind: :position,
        value: 0.5,
        opts: [velocity: 1.0]
      }

  ## Other effects

  A policy is not limited to actuator commands: `action_to_commands/3` may return
  any struct implementing `BB.Policy.Effect` (e.g. publishing a message or
  issuing a sub-command). This struct is the built-in implementation.
  """

  @enforce_keys [:path, :kind]
  defstruct [:path, :kind, :value, opts: []]

  @type kind :: :position | :velocity | :effort | :hold | :stop

  @type t :: %__MODULE__{
          path: atom() | [atom()],
          kind: kind(),
          value: number() | nil,
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

  @doc "Build a hold command for `path` (hold the current setpoint)."
  @spec hold(atom() | [atom()], keyword()) :: t()
  def hold(path, opts \\ []),
    do: %__MODULE__{path: path, kind: :hold, value: nil, opts: opts}

  @doc "Build a stop command for `path`."
  @spec stop(atom() | [atom()], keyword()) :: t()
  def stop(path, opts \\ []),
    do: %__MODULE__{path: path, kind: :stop, value: nil, opts: opts}

  @doc """
  Apply this command to `robot` via the corresponding `BB.Actuator` call.

  Returns `:ok`. Commands are delivered over PubSub (`BB.Actuator.*`) so they are
  logged and replayable; the safety gate is the caller's responsibility
  (`BB.Policy.Step` only applies effects while armed). Also reachable through the
  `BB.Policy.Effect` protocol.
  """
  @spec apply(t(), robot :: module()) :: :ok
  def apply(%__MODULE__{} = command, robot) do
    path = List.wrap(command.path)

    case command.kind do
      :position -> BB.Actuator.set_position(robot, path, command.value, command.opts)
      :velocity -> BB.Actuator.set_velocity(robot, path, command.value, command.opts)
      :effort -> BB.Actuator.set_effort(robot, path, command.value, command.opts)
      :hold -> BB.Actuator.hold(robot, path, command.opts)
      :stop -> BB.Actuator.stop(robot, path, command.opts)
    end
  end

  defimpl BB.Policy.Effect do
    defdelegate apply(command, robot), to: BB.Policy.ActuatorCommand
  end
end
