# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.NormalizerTest do
  use ExUnit.Case, async: true

  alias BB.Policy.Normalizer

  doctest BB.Policy.Normalizer

  defp close?(tensor, expected, tol \\ 1.0e-5) do
    Nx.all_close(tensor, Nx.tensor(expected), atol: tol) |> Nx.to_number() == 1
  end

  describe "new/1 validation" do
    test "accepts well-formed stats" do
      assert {:ok, %Normalizer{}} =
               Normalizer.new(
                 observation: %{s: %{strategy: :z_score, mean: 0.0, std: 1.0}},
                 action: %{a: %{strategy: :min_max, min: 0.0, max: 1.0}}
               )
    end

    test "defaults both spaces to empty" do
      assert {:ok, %Normalizer{observation: %{}, action: %{}}} = Normalizer.new([])
    end

    test "rejects a strategy that is missing required moments" do
      assert {:error, {:observation, :s, {:missing_moments, [:std]}}} =
               Normalizer.new(observation: %{s: %{strategy: :z_score, mean: 0.0}})
    end

    test "rejects an unknown strategy" do
      assert {:error, {:action, :a, {:unknown_strategy, :bogus}}} =
               Normalizer.new(action: %{a: %{strategy: :bogus}})
    end

    test "rejects stats with no strategy" do
      assert {:error, {:observation, :s, :missing_strategy}} =
               Normalizer.new(observation: %{s: %{mean: 0.0}})
    end

    test "new!/1 raises on invalid stats" do
      assert_raise ArgumentError, fn ->
        Normalizer.new!(observation: %{s: %{strategy: :z_score}})
      end
    end
  end

  describe "identity / unregistered keys" do
    test "an unregistered key passes through unchanged" do
      {:ok, n} = Normalizer.new([])
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert close?(Normalizer.normalize(n, :observation, :anything, t), [1.0, 2.0, 3.0])
      assert close?(Normalizer.denormalize(n, :action, :anything, t), [1.0, 2.0, 3.0])
    end

    test "an explicit identity strategy passes through" do
      {:ok, n} = Normalizer.new(observation: %{s: %{strategy: :identity}})
      t = Nx.tensor([5.0, -5.0])
      assert close?(Normalizer.normalize(n, :observation, :s, t), [5.0, -5.0])
    end
  end

  describe "z_score" do
    setup do
      {:ok, n} =
        Normalizer.new(observation: %{s: %{strategy: :z_score, mean: 2.0, std: 4.0}})

      %{n: n}
    end

    test "forward standardises", %{n: n} do
      # (x - 2) / 4
      result = Normalizer.normalize(n, :observation, :s, Nx.tensor([2.0, 6.0, -2.0]))
      assert close?(result, [0.0, 1.0, -1.0])
    end

    test "inverse undoes forward (round-trip)", %{n: n} do
      raw = Nx.tensor([1.5, 9.0, -3.25])
      norm = Normalizer.normalize(n, :observation, :s, raw)
      assert close?(Normalizer.denormalize(n, :observation, :s, norm), [1.5, 9.0, -3.25])
    end

    test "per-element (tensor) mean/std" do
      {:ok, n} =
        Normalizer.new(
          observation: %{s: %{strategy: :z_score, mean: [0.0, 10.0], std: [1.0, 2.0]}}
        )

      result = Normalizer.normalize(n, :observation, :s, Nx.tensor([3.0, 16.0]))
      assert close?(result, [3.0, 3.0])
    end
  end

  describe "min_max" do
    test "scales to [0, 1] by default" do
      {:ok, n} = Normalizer.new(action: %{a: %{strategy: :min_max, min: 0.0, max: 10.0}})
      result = Normalizer.normalize(n, :action, :a, Nx.tensor([0.0, 5.0, 10.0]))
      assert close?(result, [0.0, 0.5, 1.0])
    end

    test "scales to [-1, 1] with :unit_symmetric" do
      {:ok, n} =
        Normalizer.new(
          action: %{a: %{strategy: :min_max, min: -4.0, max: 4.0, range: :unit_symmetric}}
        )

      result = Normalizer.normalize(n, :action, :a, Nx.tensor([-4.0, 0.0, 4.0]))
      assert close?(result, [-1.0, 0.0, 1.0])
    end

    test "round-trips for both ranges" do
      for range <- [:unit, :unit_symmetric] do
        {:ok, n} =
          Normalizer.new(
            action: %{
              a: %{strategy: :min_max, min: [-3.14, -1.5], max: [3.14, 1.5], range: range}
            }
          )

        raw = Nx.tensor([1.2, -0.7])
        norm = Normalizer.normalize(n, :action, :a, raw)
        assert close?(Normalizer.denormalize(n, :action, :a, norm), [1.2, -0.7])
      end
    end
  end

  describe "numerical safety" do
    test "zero std does not divide by zero (constant feature -> 0)" do
      {:ok, n} =
        Normalizer.new(observation: %{s: %{strategy: :z_score, mean: 5.0, std: 0.0}})

      result = Normalizer.normalize(n, :observation, :s, Nx.tensor([5.0, 5.0]))
      assert close?(result, [0.0, 0.0])
      refute result |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    end

    test "min == max does not divide by zero (-> range minimum)" do
      {:ok, n} = Normalizer.new(action: %{a: %{strategy: :min_max, min: 3.0, max: 3.0}})
      result = Normalizer.normalize(n, :action, :a, Nx.tensor([3.0, 3.0]))
      assert close?(result, [0.0, 0.0])
      refute result |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    end
  end

  describe "stats_from_samples/3" do
    test "z_score reduces over the leading sample axis" do
      samples = Nx.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
      stats = Normalizer.stats_from_samples(samples, :z_score)
      assert stats.strategy == :z_score
      assert close?(stats.mean, [2.0, 20.0])
    end

    test "min_max captures per-element extremes and carries range" do
      samples = Nx.tensor([[1.0, -5.0], [3.0, 5.0]])
      stats = Normalizer.stats_from_samples(samples, :min_max, range: :unit_symmetric)
      assert close?(stats.min, [1.0, -5.0])
      assert close?(stats.max, [3.0, 5.0])
      assert stats.range == :unit_symmetric
    end

    test "computed stats round-trip through a normaliser" do
      samples = Nx.tensor([[0.0, 100.0], [2.0, 200.0], [4.0, 300.0]])

      {:ok, n} =
        Normalizer.new(observation: %{s: Normalizer.stats_from_samples(samples, :z_score)})

      raw = Nx.tensor([3.0, 250.0])
      norm = Normalizer.normalize(n, :observation, :s, raw)
      assert close?(Normalizer.denormalize(n, :observation, :s, norm), [3.0, 250.0])
    end
  end

  describe "load/1" do
    @json """
    {
      "observation": {
        "joint_positions": {"strategy": "z_score", "mean": [0.0, 0.1], "std": [1.0, 0.5]},
        "gripper": {"strategy": "min_max", "min": 0.0, "max": 1.0}
      },
      "action": {
        "target_positions": {"strategy": "min_max", "min": [-3.14], "max": [3.14], "range": "unit_symmetric"}
      }
    }
    """

    setup do
      path =
        Path.join(System.tmp_dir!(), "bb_policy_stats_#{System.unique_integer([:positive])}.json")

      File.write!(path, @json)
      on_exit(fn -> File.rm(path) end)
      %{path: path}
    end

    test "parses spaces, strategies, range, and tensor vs scalar moments", %{path: path} do
      assert {:ok, %Normalizer{} = n} = Normalizer.load(path)

      assert n.observation.joint_positions.strategy == :z_score
      assert %Nx.Tensor{} = n.observation.joint_positions.mean
      assert n.observation.gripper.strategy == :min_max
      assert n.observation.gripper.min == 0.0
      assert n.action.target_positions.range == :unit_symmetric

      # behaves: joint_positions[1] = (x - 0.1) / 0.5
      result = Normalizer.normalize(n, :observation, :joint_positions, Nx.tensor([0.0, 0.6]))
      assert close?(result, [0.0, 1.0])
    end

    test "returns an error for a missing file" do
      assert {:error, :enoent} = Normalizer.load("/no/such/stats.json")
    end

    test "returns an error for invalid JSON" do
      path =
        Path.join(System.tmp_dir!(), "bb_policy_bad_#{System.unique_integer([:positive])}.json")

      File.write!(path, "{not json")
      on_exit(fn -> File.rm(path) end)
      assert {:error, {:invalid_json, _}} = Normalizer.load(path)
    end
  end
end
