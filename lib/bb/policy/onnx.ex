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

    * `:model` (required) — the `.onnx` model to load, as either:
      * a path string (resolved by Ortex against the working directory), or
      * `{:priv, app, relative}` — resolved at `init/1` (runtime) against `app`'s
        priv directory via `Application.app_dir/2`. Use this when the model ships
        in a packaged app (e.g. a Nerves firmware), so the path is correct on the
        device rather than frozen to the build host. A missing file fails `init/1`
        with `{:error, {:model_not_found, path}}`.
    * `:observation` (required) — an ordered list describing how to build the
      model's input vector from robot state. Each entry is `{source, joints}` or
      `{source, joints, opts}`, where `source` is `:positions` or `:velocities`,
      `joints` is the ordered list of joint names to read, and `opts` may carry a
      `:key` (the normalisation feature key, default `:observation`). Entries are
      read in order; entries sharing a `:key` are normalised together (their
      stats cover the combined vector); the normalised groups are concatenated in
      the order each key first appears and reshaped to `[1, N]`. (A bare keyword
      list like `[positions: joints]` still works — it's `{:positions, joints}`.)
    * `:action` (required) — an ordered list of `{joints, kind}` or
      `{joints, kind, opts}` mapping the model's output columns to actuator
      commands, where `kind` is `:position`, `:velocity`, or `:effort` and `opts`
      may carry a `:key` (default `:action`). Columns are consumed left-to-right;
      entries sharing a `:key` are denormalised together; the joint name doubles
      as the actuator path (wrap in a list for a nested path).
    * `:normalizer` — a `BB.Policy.Normalizer` struct, or a path to its JSON.
      Observation entries are normalised, and action entries denormalised, under
      their `:key` (default `:observation`/`:action`). Use explicit keys to match
      a per-feature export, e.g. a LeRobot `observation.state` / `action` JSON:
      `observation: [{:positions, joints, key: :"observation.state"}]`. **Every
      key the specs reference must have registered stats — `init/1` fails with
      `{:error, {:missing_normalizer_stats, …}}` otherwise** (a missing key would
      otherwise mean silent unnormalised inference). Optional; when omitted, an
      explicit all-identity normaliser is built for the specs' keys.
      > LeRobot `MIN_MAX` maps to `[-1, 1]`, so a LeRobot min-max export wants
      > `"range": "unit_symmetric"` in its stats (see `BB.Policy.Normalizer`).
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
  @type entry_opts :: [{:key, atom()}]
  @type observation_spec :: [{source(), [atom()]} | {source(), [atom()], entry_opts()}]
  @type action_spec :: [
          {atom() | [atom()], ActuatorCommand.kind()}
          | {atom() | [atom()], ActuatorCommand.kind(), entry_opts()}
        ]

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
         {:ok, model_spec} <- fetch(opts, :model),
         {:ok, model_path} <- resolve_model_path(model_spec),
         {:ok, observation} <- fetch(opts, :observation),
         {:ok, action} <- fetch(opts, :action),
         {:ok, normalizer} <- load_normalizer(opts[:normalizer], observation, action),
         :ok <- validate_normalizer_keys(normalizer, observation, action) do
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
    # Read each entry's raw values, then normalise per feature key: entries
    # sharing a `:key` are concatenated and normalised together (their stats
    # cover the combined vector), and the normalised groups are concatenated in
    # the order each key first appears. The model input column order is that
    # group order.
    input =
      state.observation
      |> Enum.map(fn entry ->
        {source, joints, key} = observation_entry(entry)
        {key, read_joints(robot_state, source, joints)}
      end)
      |> group_by_key()
      |> Enum.map(fn {key, values} ->
        Normalizer.normalize(state.normalizer, :observation, key, Nx.tensor(values, type: :f32))
      end)
      |> Nx.concatenate()

    {%{input: input}, state}
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
    values = denormalise_action(state.normalizer, state.action, row)

    {commands, _rest} =
      Enum.reduce(state.action, {[], values}, fn entry, {acc, remaining} ->
        {joints, kind, _key} = action_entry(entry)
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

  # Resolve the :model option to a filesystem path, at init (runtime) so it lands
  # on whatever node we're on — important for a Nerves device, where the install
  # root differs from the build host and a path frozen at compile time is wrong.
  #
  #   * a binary path is used verbatim (Ortex resolves it against the cwd);
  #   * `{:priv, app, relative}` resolves against `app`'s priv dir on this node
  #     via Application.app_dir/2 (the right idiom for a packaged model file).
  #
  # A missing file fails init cleanly rather than crashing in Ortex.load.
  defp resolve_model_path({:priv, app, relative}) when is_atom(app) and is_binary(relative) do
    check_model_file(Application.app_dir(app, Path.join("priv", relative)))
  end

  defp resolve_model_path(path) when is_binary(path), do: check_model_file(path)

  defp resolve_model_path(other), do: {:error, {:invalid_model, other}}

  defp check_model_file(path) do
    if File.exists?(path), do: {:ok, path}, else: {:error, {:model_not_found, path}}
  end

  # No normalizer given → build an explicit all-identity one covering every key
  # the specs reference, so "no normalisation" is a deliberate identity rather
  # than a silently-missing key.
  defp load_normalizer(nil, observation, action) do
    obs_keys = observation |> Enum.map(&elem(observation_entry(&1), 2)) |> Enum.uniq()
    action_keys = action |> Enum.map(&elem(action_entry(&1), 2)) |> Enum.uniq()
    identity = fn keys -> Map.new(keys, &{&1, %{strategy: :identity}}) end

    Normalizer.new(observation: identity.(obs_keys), action: identity.(action_keys))
  end

  defp load_normalizer(%Normalizer{} = normalizer, _observation, _action), do: {:ok, normalizer}

  defp load_normalizer(path, _observation, _action) when is_binary(path),
    do: Normalizer.load(path)

  # An observation entry is `{source, joints}` (key defaults to :observation) or
  # `{source, joints, opts}` carrying an explicit `key:` (e.g. a LeRobot feature
  # name like :"observation.state").
  defp observation_entry({source, joints}), do: {source, joints, :observation}

  defp observation_entry({source, joints, opts}),
    do: {source, joints, Keyword.get(opts, :key, :observation)}

  # An action entry is `{joints, kind}` (key defaults to :action) or
  # `{joints, kind, opts}` with an explicit `key:`.
  defp action_entry({joints, kind}), do: {joints, kind, :action}
  defp action_entry({joints, kind, opts}), do: {joints, kind, Keyword.get(opts, :key, :action)}

  # Concatenate the values of entries sharing a key, preserving the order each
  # key first appears. Returns `[{key, [value]}]`.
  defp group_by_key(keyed_values) do
    keyed_values
    |> Enum.reduce({[], %{}}, fn {key, values}, {order, acc} ->
      order = if Map.has_key?(acc, key), do: order, else: [key | order]
      {order, Map.update(acc, key, values, &(&1 ++ values))}
    end)
    |> then(fn {order, acc} -> Enum.map(Enum.reverse(order), &{&1, acc[&1]}) end)
  end

  # Denormalise the flat action row per feature key, then flatten back to a list
  # of engineering-unit values in the original entry/column order. Entries
  # sharing a key are denormalised together (their stats cover the combined
  # action vector).
  defp denormalise_action(normalizer, action_spec, row) do
    flat = Nx.to_flat_list(row)

    # [{key, width}] per entry, in column order.
    widths =
      Enum.map(action_spec, fn entry ->
        {joints, _kind, key} = action_entry(entry)
        {key, length(List.wrap(joints))}
      end)

    # Slice the row into per-entry chunks, group by key, denormalise each group,
    # then re-slice into per-entry values so the caller can map them to joints.
    {entry_slices, _} =
      Enum.map_reduce(widths, flat, fn {key, width}, remaining ->
        {chunk, rest} = Enum.split(remaining, width)
        {{key, chunk}, rest}
      end)

    denorm_by_key =
      entry_slices
      |> group_by_key()
      |> Map.new(fn {key, values} ->
        denormalised =
          Normalizer.denormalize(normalizer, :action, key, Nx.tensor(values, type: :f32))

        {key, Nx.to_flat_list(denormalised)}
      end)

    # Re-emit per-entry values in column order, drawing from each key's
    # denormalised buffer in first-appearance order.
    {result, _} =
      Enum.flat_map_reduce(entry_slices, denorm_by_key, fn {key, chunk}, buffers ->
        {taken, rest} = Enum.split(buffers[key], length(chunk))
        {taken, Map.put(buffers, key, rest)}
      end)

    result
  end

  # Validate, at init, that every feature key the specs reference has registered
  # normalisation stats. Turns a missing-stats mistake into a clear setup-time
  # error rather than a run-time raise mid-episode (or, worse, silent passthrough).
  defp validate_normalizer_keys(normalizer, observation, action) do
    obs_keys = observation |> Enum.map(&elem(observation_entry(&1), 2)) |> Enum.uniq()
    action_keys = action |> Enum.map(&elem(action_entry(&1), 2)) |> Enum.uniq()

    missing =
      (Enum.map(obs_keys, &{:observation, &1}) ++ Enum.map(action_keys, &{:action, &1}))
      |> Enum.reject(fn {space, key} -> Normalizer.has_key?(normalizer, space, key) end)

    case missing do
      [] -> :ok
      _ -> {:error, {:missing_normalizer_stats, missing}}
    end
  end

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
