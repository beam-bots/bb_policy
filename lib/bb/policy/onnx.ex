# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.ONNX do
  @moduledoc """
  `BB.Policy` implementation that loads ONNX models via [Ortex](https://github.com/elixir-nx/ortex).

  This is the recommended way to deploy policies trained in Python (PyTorch,
  JAX, TensorFlow) on Beam Bots. The model is loaded once in `c:BB.Policy.init/1`
  and inference runs on each control tick.

  ## Usage

      BB.Policy.run(MyRobot, BB.Policy.ONNX, %{task: :pick_mug},
        policy_opts: [
          model: "priv/models/pick_mug.onnx",
          normalizer: "priv/models/pick_mug.json",
          observation: [positions: [:shoulder, :elbow, :wrist]],
          action: [{[:shoulder, :elbow, :wrist], :position}]
        ],
        rate_hz: 20
      )

  ## Options

    * `:model` (required) — path to the `.onnx` file.
    * `:observation` (required) — an ordered keyword list describing how to build
      the model's single input vector from robot state. Each entry is
      `source: joints`, where `source` is `:positions` or `:velocities` and
      `joints` is the ordered list of joint names to read. Entries are
      concatenated in order, normalised (see `:normalizer`), and reshaped to
      `[1, N]`.
    * `:action` (required) — an ordered list of `{joints, kind}` mapping the
      model's output columns to actuator commands, where `kind` is `:position`,
      `:velocity`, or `:effort`. Columns are consumed left-to-right across the
      entries; the joint name doubles as the actuator path (wrap in a list for a
      nested path).
    * `:normalizer` — a `BB.Policy.Normalizer` struct, or a path to its JSON.
      Observations are normalised under key `:observation`, actions
      denormalised under key `:action`. Optional (defaults to identity).
    * `:execution_providers` — ordered Ortex execution-provider list. Default
      `[:cpu]`. Note ort silently falls back to CPU if a provider isn't compiled
      in (see PROJECT_PLAN R4).
    * `:temporal_ensemble_coeff` — when set to a number, selects temporal
      ensembling instead of the receding-horizon queue (see "Action chunking").
      Larger values weight the most recent chunk more heavily.

  ## Inference: `Ortex.run/2`, not batched serving

  For a single robot at a fixed control rate, inference is one call at a time.
  This implementation calls `Ortex.run/2` directly. `Nx.Serving` batched
  execution exists to amortise overhead across *concurrent* requests; a single
  20 Hz loop never fills a batch, and `batch_timeout` would only add latency.
  Batched serving is reserved for a future multi-camera/multi-policy path.

  ## Exporting from LeRobot — read this before assuming it "just works"

  There is no first-class one-line ONNX export in LeRobot. In particular:

    * `policy.select_action` contains Python control flow (the action-chunk
      queue, temporal ensembling) that is **not traceable** — you export
      inference-only subgraphs wrapped in thin `nn.Module`s, often splitting
      vision encoder and transformer into separate graphs with static shapes.
    * **Normalisation is stripped from the graph.** Export the dataset
      statistics separately and apply them with `BB.Policy.Normalizer`.
    * **Diffusion / VLA (π0) policies** have iterative denoising loops that do
      not export cleanly today; target **ACT first**.

  ## Action chunking (ACT)

  ACT predicts a horizon of future actions per inference. Two regimes:

    * **Receding-horizon queue** (default) — each `c:BB.Policy.act/2` pops one
      action row; when the queue empties it runs one inference and refills from
      the predicted chunk. Cheaper, fewer inferences. A model that outputs a
      single action (`[1, action_dim]`) is a chunk of length one — inference
      every tick.
    * **Temporal ensembling** (`:temporal_ensemble_coeff` set) — infer every
      tick; for the current timestep, blend the predictions of all overlapping
      chunks with exponential weights `wᵢ = exp(-coeff · age)`. Smoother, more
      compute. Stale chunks (whose horizon no longer covers the next step) are
      dropped.

  > #### Optional dependency {: .info}
  >
  > `ortex` builds a Rust NIF and downloads an onnxruntime binary, so it is an
  > optional dependency gated behind `ORTEX=1` in this repo. `init/1` returns a
  > clear error if Ortex is not loaded.
  """

  @behaviour BB.Policy

  alias BB.Policy.ActuatorCommand
  alias BB.Policy.Normalizer
  alias BB.Robot.State, as: RobotState

  defstruct [
    :model,
    :normalizer,
    :observation,
    :action,
    :ensemble_coeff,
    action_queue: [],
    chunks: [],
    step: 0
  ]

  @type source :: :positions | :velocities
  @type observation_spec :: [{source(), [atom()]}]
  @type action_spec :: [{atom() | [atom()], ActuatorCommand.kind()}]

  @type t :: %__MODULE__{
          model: term(),
          normalizer: Normalizer.t(),
          observation: observation_spec(),
          action: action_spec(),
          ensemble_coeff: float() | nil,
          action_queue: [Nx.Tensor.t()],
          chunks: [{non_neg_integer(), Nx.Tensor.t()}],
          step: non_neg_integer()
        }

  @impl BB.Policy
  def init(opts) do
    with :ok <- ensure_ortex(),
         {:ok, model_path} <- fetch(opts, :model),
         {:ok, observation} <- fetch(opts, :observation),
         {:ok, action} <- fetch(opts, :action),
         {:ok, normalizer} <- load_normalizer(opts[:normalizer]) do
      # apply/3 rather than a direct call: ortex is an optional dependency, so
      # the module may be absent at compile time. ensure_ortex/0 above guarantees
      # it is loaded before we reach here.
      providers = Keyword.get(opts, :execution_providers, [:cpu])
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      model = apply(Ortex, :load, [model_path, providers])

      {:ok,
       %__MODULE__{
         model: model,
         normalizer: normalizer,
         observation: observation,
         action: action,
         ensemble_coeff: opts[:temporal_ensemble_coeff]
       }}
    end
  end

  @impl BB.Policy
  def reset(%__MODULE__{} = state), do: %{state | action_queue: [], chunks: [], step: 0}

  @impl BB.Policy
  def observe(robot_state, _sensors, %__MODULE__{} = state) do
    vector =
      state.observation
      |> Enum.flat_map(fn {source, joints} -> read_joints(robot_state, source, joints) end)
      |> Nx.tensor(type: :f32)

    normalised = Normalizer.normalize(state.normalizer, :observation, :input, vector)
    {%{input: normalised}, state}
  end

  # Temporal ensembling: infer every tick, then blend every stored chunk's
  # prediction for the current absolute step with exponential weights
  # w_i = exp(-coeff * age). Smoother, more compute (PROJECT_PLAN D5/§6.5).
  @impl BB.Policy
  def act(%{input: input}, %__MODULE__{ensemble_coeff: coeff} = state) when is_number(coeff) do
    chunk = run_inference(state.model, input)
    chunks = [{state.step, chunk} | state.chunks]
    {action, chunks} = ensemble(chunks, state.step, coeff)
    {%{action: action}, %{state | chunks: chunks, step: state.step + 1}}
  end

  # Receding-horizon queue (default): execute the predicted chunk one row per
  # tick; infer again only when the queue empties. Cheaper, fewer inferences.
  def act(%{input: input}, %__MODULE__{action_queue: []} = state) do
    [row | rest] = rows(run_inference(state.model, input))
    {%{action: row}, %{state | action_queue: rest}}
  end

  def act(_observation, %__MODULE__{action_queue: [row | rest]} = state) do
    {%{action: row}, %{state | action_queue: rest}}
  end

  @impl BB.Policy
  def action_to_commands(%{action: row}, _robot, %__MODULE__{} = state) do
    denormalised = Normalizer.denormalize(state.normalizer, :action, :output, row)
    values = Nx.to_flat_list(denormalised)

    {commands, _rest} =
      Enum.reduce(state.action, {[], values}, fn {joints, kind}, {acc, remaining} ->
        joints = List.wrap(joints)
        {taken, rest} = Enum.split(remaining, length(joints))

        cmds =
          joints
          |> Enum.zip(taken)
          |> Enum.map(fn {joint, value} -> build_command(kind, joint, value) end)

        {acc ++ cmds, rest}
      end)

    {:ok, commands}
  rescue
    error -> {:error, {:action_to_commands, error}}
  end

  # --- internals -----------------------------------------------------------

  defp ensure_ortex do
    if Code.ensure_loaded?(Ortex) do
      :ok
    else
      {:error, :ortex_not_available}
    end
  end

  defp fetch(opts, key) do
    case Keyword.fetch(opts, key) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, {:missing_option, key}}
    end
  end

  defp load_normalizer(nil), do: {:ok, %Normalizer{}}
  defp load_normalizer(%Normalizer{} = normalizer), do: {:ok, normalizer}
  defp load_normalizer(path) when is_binary(path), do: Normalizer.load(path)

  defp read_joints(robot_state, :positions, joints) do
    all = RobotState.get_all_positions(robot_state)
    Enum.map(joints, &Map.get(all, &1, 0.0))
  end

  defp read_joints(robot_state, :velocities, joints) do
    all = RobotState.get_all_velocities(robot_state)
    Enum.map(joints, &Map.get(all, &1, 0.0))
  end

  defp run_inference(model, input) do
    batched = Nx.reshape(input, {1, Nx.size(input)})
    # apply/3: see the note in init/1 — Ortex is an optional dependency.
    # credo:disable-for-next-line Credo.Check.Refactor.Apply
    {output} = apply(Ortex, :run, [model, {batched}])
    Nx.backend_transfer(output)
  end

  # Normalise the model output into a list of per-step action rows (1-D tensors),
  # whether the model emits [1, action_dim] or a chunk [1, chunk, action_dim].
  defp rows(output) do
    case Nx.shape(output) do
      {1, _dim} -> [output[0]]
      {1, chunk, _dim} -> for i <- 0..(chunk - 1), do: output[0][i]
      {_dim} -> [output]
    end
  end

  # Blend every chunk's prediction for absolute step `step` with exponential
  # weights decaying by chunk age, then drop chunks that no longer cover `step`.
  defp ensemble(chunks, step, coeff) do
    contributions =
      for {start, chunk} <- chunks,
          rows = rows(chunk),
          (idx = step - start) < length(rows),
          idx >= 0 do
        {Enum.at(rows, idx), :math.exp(-coeff * idx)}
      end

    weighted =
      contributions
      |> Enum.map(fn {row, w} -> Nx.multiply(row, w) end)
      |> Enum.reduce(&Nx.add/2)

    total = contributions |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    action = Nx.divide(weighted, total)

    # Keep only chunks whose horizon still covers the *next* step.
    live =
      for {start, chunk} <- chunks, step + 1 - start < length(rows(chunk)), do: {start, chunk}

    {action, live}
  end

  defp build_command(:position, joint, value), do: ActuatorCommand.position(joint, value)
  defp build_command(:velocity, joint, value), do: ActuatorCommand.velocity(joint, value)
  defp build_command(:effort, joint, value), do: ActuatorCommand.effort(joint, value)
end
