# SPDX-FileCopyrightText: 2026 James Harton <james@harton.nz>
# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy do
  @moduledoc """
  Behaviour for learned policies that map observations to actions.

  A *policy* is a function `π: observation → action`. Given what the robot
  perceives (joint positions, velocities, camera frames, force readings), the
  policy produces what it should do (target positions, velocities, gripper
  commands). Rather than programming every motion explicitly, operators train
  policies from demonstrations or simulation and deploy them on real hardware
  with full safety-system integration.

  Policies are *stateful*. Some architectures (RNNs, transformers with context,
  and action-chunking models such as ACT) carry hidden state or a queue of
  pending actions across timesteps. This behaviour threads an opaque `state/0`
  through every callback so implementations own their own lifecycle.

  ## Lifecycle

  `c:init/1` is called once when the policy loads. Then, on every control tick,
  `BB.Policy.Runner` drives this cycle:

      observe/3  →  act/2  →  action_to_commands/3

  `c:reset/1` is called at episode boundaries to clear accumulated state.

  ## Implementing a policy

      defmodule MyRobot.Policy.PickMug do
        @behaviour BB.Policy

        @impl BB.Policy
        def init(opts), do: {:ok, %{model: load(opts[:model])}}

        @impl BB.Policy
        def reset(state), do: %{state | hidden: nil}

        # ...observe/3, act/2, action_to_commands/3
      end

  See `BB.Policy.ONNX` for the recommended implementation that loads models
  trained in Python (LeRobot, PyTorch, JAX) and exported to ONNX.

  > #### Safety {: .warning}
  >
  > Policies do not bypass the safety system. `BB.Policy.Runner` checks
  > `BB.Safety.armed?/1` before applying any command, and the commands a policy
  > emits are still subject to the robot's joint and velocity limits. A policy
  > that returns bad actions cannot drive hardware outside its configured
  > envelope. See `BB.Policy.Runner` for details.
  """

  @typedoc """
  A named bundle of observation tensors, normalised and ready for inference.

  Keys are operator-chosen names (e.g. `:joint_positions`, `:camera`) matching
  the policy's `observation_keys`.
  """
  @type observation :: %{atom() => Nx.Tensor.t()}

  @typedoc """
  A named bundle of action tensors produced by inference, prior to
  denormalisation and conversion to actuator commands.

  For action-chunking policies a single `act/2` may produce a whole horizon of
  actions; how that is unrolled into per-tick commands is the implementation's
  concern (see `t:state/0`).
  """
  @type action :: %{atom() => Nx.Tensor.t()}

  @typedoc "Opaque, implementation-private policy state threaded through callbacks."
  @type state :: term()

  @typedoc "Goal/task specification passed to the runner, forwarded to the policy via `t:state/0`."
  @type goal :: term()

  @doc """
  Initialise the policy.

  Called once when the policy is loaded. Loads model weights, sets up
  normalisation, validates configuration, and returns the initial state.

  `opts` are the `:policy_opts` configured on the runner.
  """
  @callback init(opts :: keyword()) :: {:ok, state()} | {:error, term()}

  @doc """
  Reset policy state at an episode boundary.

  Clears accumulated hidden state and any pending action chunk. For stateless
  policies this is the identity function.
  """
  @callback reset(state()) :: state()

  @doc """
  Construct the observation from robot state and sensor data.

  Reads the quantities the policy needs (joint positions/velocities from
  `robot_state`, latest sensor frames from `sensors`), normalises them, and
  returns the observation tensors. May update `state` (e.g. to track an
  observation history buffer).

  `robot_state` is the value returned by `BB.Robot.Runtime.get_robot_state/1`;
  `sensors` is a map of the most recent sensor payloads keyed by sensor name.
  """
  @callback observe(
              robot_state :: term(),
              sensors :: %{atom() => term()},
              state()
            ) :: {observation(), state()}

  @doc """
  Run inference to produce an action.

  Given an observation, returns the action to execute and the updated state.
  Recurrent and chunking policies update their hidden state / action queue here.

  A policy that has reached its goal returns `{:done, state}` instead of an
  action; `BB.Policy.Runner` then ends the episode with reason `:completed`.
  """
  @callback act(observation(), state()) :: {action() | :done, state()}

  @doc """
  Convert an action into actuator commands.

  Denormalises the action and builds `BB.Policy.ActuatorCommand` structs. The
  runner applies the returned commands (subject to safety) on this tick.
  """
  @callback action_to_commands(action(), robot :: module(), state()) ::
              {:ok, [BB.Policy.ActuatorCommand.t()]} | {:error, term()}

  @doc """
  Return policy metadata for introspection (architecture, input/output spec, …).

  Optional.
  """
  @callback info(state()) :: map()

  @optional_callbacks [info: 1]

  @doc """
  Run a policy on `robot` until completion, timeout, or safety intervention.

  This is the package's imperative entry point. It delegates to
  `BB.Policy.Runner.run/4`; see that function for options.

      {:ok, result} =
        BB.Policy.run(MyRobot, BB.Policy.ONNX, %{task: :pick_mug},
          policy_opts: [model: "priv/models/pick_mug.onnx"],
          rate_hz: 20,
          timeout: :timer.seconds(30)
        )
  """
  @spec run(robot :: module(), policy :: module(), goal(), keyword()) ::
          {:ok, term()} | {:error, term()}
  defdelegate run(robot, policy_module, goal, opts \\ []), to: BB.Policy.Runner
end
