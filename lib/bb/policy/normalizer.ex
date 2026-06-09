# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.Normalizer do
  @moduledoc """
  Observation and action normalisation.

  Policies expect normalised inputs and produce normalised outputs. The
  statistics used for normalisation are a property of the *training dataset*
  and must be applied identically at inference time — frameworks such as
  LeRobot strip normalisation out of the exported ONNX graph and ship the
  statistics separately, so the runtime owns this step.

  A normaliser holds statistics for two spaces — `:observation` and `:action` —
  each a map of *key* to per-key statistics. Keys match the policy's
  `observation_keys`/`action_keys`. Each key carries its own strategy, so a
  policy can z-score its joint observations while min-max scaling its actions.

  ## Strategies

    * `:z_score` — standardise to mean `0`, std `1`: `(x - mean) / std`.
      Requires `:mean` and `:std`.
    * `:min_max` — scale a known range into `[0, 1]` (default) or `[-1, 1]`:
      `(x - min) / (max - min)`, optionally rescaled. Requires `:min` and `:max`.
    * `:identity` — passthrough. Requires no statistics.

  The observation path calls `normalize/4` (raw reading → policy input); the
  action path calls `denormalize/4` (policy output → engineering units). The two
  are exact inverses for a given key.

  ## Statistics

  Per-key stats are a map carrying a `:strategy` plus the moments that strategy
  needs. Moment values may be numbers or `t:Nx.Tensor.t/0` (for per-element
  stats, e.g. per-joint or per-channel) and are broadcast against the input
  tensor.

      %{
        observation: %{
          joint_positions: %{strategy: :z_score, mean: [0.0, 0.1], std: [1.0, 0.5]},
          camera: %{strategy: :min_max, min: 0.0, max: 255.0}
        },
        action: %{
          target_positions: %{strategy: :min_max, min: [-3.14, -1.5], max: [3.14, 1.5], range: :unit_symmetric}
        }
      }

  Build one with `new/1`, or load the JSON a training pipeline exports with
  `load/1`.

  ## Numerical safety

  Degenerate statistics (zero std, or `min == max`) would divide by zero. Such
  elements are treated as having unit scale, so a constant feature normalises to
  `0` (z-score) or the range minimum (min-max) rather than `NaN`/`Inf`.
  """

  @type space :: :observation | :action
  @type strategy :: :z_score | :min_max | :identity
  @type moment :: number() | [number()] | Nx.Tensor.t()

  @typedoc """
  Per-key statistics: a `:strategy` plus the moments it needs.

  For `:min_max`, the optional `:range` selects the output interval:
  `:unit` (default, `[0, 1]`) or `:unit_symmetric` (`[-1, 1]`).
  """
  @type key_stats :: %{
          required(:strategy) => strategy(),
          optional(:mean) => moment(),
          optional(:std) => moment(),
          optional(:min) => moment(),
          optional(:max) => moment(),
          optional(:range) => :unit | :unit_symmetric
        }

  @type stats :: %{atom() => key_stats()}

  @type t :: %__MODULE__{
          observation: stats(),
          action: stats()
        }

  defstruct observation: %{}, action: %{}

  @epsilon 1.0e-8

  @doc """
  Build a normaliser from observation and/or action statistics.

  Accepts a keyword list or map with `:observation` and `:action` keys, each a
  map of `key => t:key_stats/0`. Validates that each key's statistics carry the
  moments its strategy requires; returns `{:error, reason}` otherwise.

  ## Examples

      iex> {:ok, n} = BB.Policy.Normalizer.new(
      ...>   observation: %{state: %{strategy: :z_score, mean: 0.0, std: 2.0}}
      ...> )
      iex> n.observation.state.strategy
      :z_score
  """
  @spec new(keyword() | map()) :: {:ok, t()} | {:error, term()}
  def new(spec) do
    spec = Map.new(spec)
    observation = Map.get(spec, :observation, %{})
    action = Map.get(spec, :action, %{})

    with :ok <- validate_space(observation, :observation),
         :ok <- validate_space(action, :action) do
      {:ok, %__MODULE__{observation: observation, action: action}}
    end
  end

  @doc """
  Like `new/1` but raises `ArgumentError` on invalid statistics.
  """
  @spec new!(keyword() | map()) :: t()
  def new!(spec) do
    case new(spec) do
      {:ok, normalizer} -> normalizer
      {:error, reason} -> raise ArgumentError, "invalid normaliser: #{inspect(reason)}"
    end
  end

  @doc """
  Normalise `tensor` for `key` in `space` (`:observation` or `:action`).

  A key with no registered statistics is passed through unchanged, so a
  normaliser only needs entries for the keys it actually scales.
  """
  @spec normalize(t(), space(), atom(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def normalize(%__MODULE__{} = normalizer, space, key, tensor) do
    apply_stats(fetch_stats(normalizer, space, key), :forward, tensor)
  end

  @doc """
  Invert `normalize/4`: map a normalised value for `key` back to engineering units.
  """
  @spec denormalize(t(), space(), atom(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def denormalize(%__MODULE__{} = normalizer, space, key, tensor) do
    apply_stats(fetch_stats(normalizer, space, key), :inverse, tensor)
  end

  @doc """
  Compute statistics from a sample tensor.

  Convenience for hardcoding stats from a recorded batch. `tensor`'s leading
  axis is the sample axis; statistics are reduced over it (so per-element stats
  are retained). `strategy` selects which moments to compute.

  ## Examples

      iex> samples = Nx.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
      iex> stats = BB.Policy.Normalizer.stats_from_samples(samples, :min_max)
      iex> Nx.to_flat_list(stats.min)
      [0.0, 10.0]
  """
  @spec stats_from_samples(Nx.Tensor.t(), strategy(), keyword()) :: key_stats()
  def stats_from_samples(tensor, strategy, opts \\ [])

  def stats_from_samples(tensor, :z_score, _opts) do
    %{
      strategy: :z_score,
      mean: Nx.mean(tensor, axes: [0]),
      std: Nx.standard_deviation(tensor, axes: [0])
    }
  end

  def stats_from_samples(tensor, :min_max, opts) do
    %{
      strategy: :min_max,
      min: Nx.reduce_min(tensor, axes: [0]),
      max: Nx.reduce_max(tensor, axes: [0]),
      range: Keyword.get(opts, :range, :unit)
    }
  end

  def stats_from_samples(_tensor, :identity, _opts), do: %{strategy: :identity}

  @doc """
  Load a normaliser from a JSON statistics file produced by a training export.

  The file is a JSON object with optional `"observation"` and `"action"`
  objects, each mapping a key name to a stats object:

      {
        "observation": {
          "joint_positions": {"strategy": "z_score", "mean": [0.0, 0.1], "std": [1.0, 0.5]}
        },
        "action": {
          "target_positions": {"strategy": "min_max", "min": [-3.14], "max": [3.14], "range": "unit_symmetric"}
        }
      }

  List-valued moments become `Nx` tensors; scalar moments stay scalar. Uses the
  Elixir standard-library `JSON` module (no extra dependency).
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    with {:ok, contents} <- File.read(path),
         {:ok, decoded} <- decode_json(contents),
         {:ok, observation} <- parse_space(decoded, "observation"),
         {:ok, action} <- parse_space(decoded, "action") do
      new(observation: observation, action: action)
    end
  end

  # --- internals -----------------------------------------------------------

  defp fetch_stats(normalizer, :observation, key), do: Map.get(normalizer.observation, key)
  defp fetch_stats(normalizer, :action, key), do: Map.get(normalizer.action, key)

  # No stats registered for this key, or an explicit identity → passthrough.
  defp apply_stats(nil, _direction, tensor), do: tensor
  defp apply_stats(%{strategy: :identity}, _direction, tensor), do: tensor

  defp apply_stats(%{strategy: :z_score} = stats, direction, tensor) do
    mean = to_tensor(stats.mean)
    std = safe_scale(to_tensor(stats.std))

    case direction do
      :forward -> Nx.divide(Nx.subtract(tensor, mean), std)
      :inverse -> Nx.add(Nx.multiply(tensor, std), mean)
    end
  end

  defp apply_stats(%{strategy: :min_max} = stats, direction, tensor) do
    min = to_tensor(stats.min)
    max = to_tensor(stats.max)
    span = safe_scale(Nx.subtract(max, min))
    {lo, hi} = range_bounds(Map.get(stats, :range, :unit))

    case direction do
      :forward ->
        # raw -> [0,1] -> [lo,hi]
        unit = Nx.divide(Nx.subtract(tensor, min), span)
        Nx.add(Nx.multiply(unit, hi - lo), lo)

      :inverse ->
        # [lo,hi] -> [0,1] -> raw
        unit = Nx.divide(Nx.subtract(tensor, lo), hi - lo)
        Nx.add(Nx.multiply(unit, span), min)
    end
  end

  defp range_bounds(:unit), do: {0.0, 1.0}
  defp range_bounds(:unit_symmetric), do: {-1.0, 1.0}

  # Replace ~zero scales with 1.0 so constant features don't divide by zero.
  defp safe_scale(scale) do
    Nx.select(Nx.less(Nx.abs(scale), @epsilon), Nx.tensor(1.0, type: Nx.type(scale)), scale)
  end

  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(value) when is_number(value), do: Nx.tensor(value, type: :f32)
  defp to_tensor(value) when is_list(value), do: Nx.tensor(value, type: :f32)

  # --- validation ----------------------------------------------------------

  @required_moments %{
    z_score: [:mean, :std],
    min_max: [:min, :max],
    identity: []
  }

  defp validate_space(space_stats, space_name) when is_map(space_stats) do
    Enum.reduce_while(space_stats, :ok, fn {key, stats}, :ok ->
      case validate_key_stats(stats) do
        :ok -> {:cont, :ok}
        {:error, reason} -> {:halt, {:error, {space_name, key, reason}}}
      end
    end)
  end

  defp validate_space(_other, space_name), do: {:error, {space_name, :not_a_map}}

  defp validate_key_stats(%{strategy: strategy} = stats)
       when is_map_key(@required_moments, strategy) do
    missing = Enum.reject(@required_moments[strategy], &Map.has_key?(stats, &1))

    case missing do
      [] -> :ok
      _ -> {:error, {:missing_moments, missing}}
    end
  end

  defp validate_key_stats(%{strategy: strategy}), do: {:error, {:unknown_strategy, strategy}}
  defp validate_key_stats(_stats), do: {:error, :missing_strategy}

  # --- JSON parsing --------------------------------------------------------

  defp decode_json(contents) do
    {:ok, JSON.decode!(contents)}
  rescue
    error -> {:error, {:invalid_json, error}}
  end

  defp parse_space(decoded, name) do
    case Map.get(decoded, name, %{}) do
      map when is_map(map) ->
        {:ok, Map.new(map, fn {key, stats} -> {String.to_atom(key), parse_key_stats(stats)} end)}

      _other ->
        {:error, {:invalid_space, name}}
    end
  end

  defp parse_key_stats(stats) when is_map(stats) do
    Map.new(stats, &parse_key_stat/1)
  end

  defp parse_key_stat({"strategy", value}), do: {:strategy, String.to_atom(value)}
  defp parse_key_stat({"range", value}), do: {:range, String.to_atom(value)}

  defp parse_key_stat({moment, value}) when moment in ["mean", "std", "min", "max"] do
    {String.to_atom(moment), parse_moment(value)}
  end

  defp parse_key_stat({other, value}), do: {String.to_atom(other), value}

  defp parse_moment(value) when is_list(value), do: Nx.tensor(value, type: :f32)
  defp parse_moment(value) when is_number(value), do: value
end
