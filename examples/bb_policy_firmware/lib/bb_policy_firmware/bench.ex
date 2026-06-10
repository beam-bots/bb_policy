# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BbPolicyFirmware.Bench do
  @moduledoc """
  The on-device end-to-end harness. Run `BbPolicyFirmware.Bench.run/0` from the
  device console (over `ssh` / IEx) after booting the firmware.

  It runs three checks and prints a verdict for each:

    1. **inference** — load the bundled ONNX model with `BB.Policy.ONNX` and run
       one `act/2`, asserting the exact expected output. Proves `Ortex.run/2`
       executes on the Pi's ARM CPU (the hardest bit, R1).
    2. **loop** — drive the simulated robot through `BB.Policy.run/4`, asserting
       the episode completes. Proves the full policy→actuator loop, with the
       safety gate, on-device.
    3. **latency** — time many `act/2` calls; report p50/p99 and whether the
       20 Hz (50 ms) control budget holds. Watch **p99**, not the mean — the
       Rust NIF can stall a scheduler (R3).
  """

  alias BB.Policy.ONNX

  @robot BbPolicyFirmware.Robot
  @expected [4.5, 6.5]

  # Resolve the model path at RUNTIME — Application.app_dir/2 returns the install
  # path on whatever node we're on (the device's /srv/erlang/..., not the build
  # host's). A module attribute would freeze the build host's path into the beam.
  defp model_path,
    do: Application.app_dir(:bb_policy_firmware, "priv/models/linear_policy.onnx")

  # Same fixture as the dev-box test: action = obs @ W + b, so [1,2,3] -> [4.5, 6.5].
  defp policy_opts do
    [
      model: model_path(),
      observation: [positions: [:a, :b, :c]],
      action: [{[:a, :b, :c], :position}]
    ]
  end

  @doc "Run all three checks and print a summary."
  def run do
    IO.puts("\n=== bb_policy on-device end-to-end ===")
    IO.puts("model: #{model_path()}")
    inference = check_inference()
    loop = check_loop()
    latency = check_latency(200, 20)

    IO.puts("\n--- summary ---")
    IO.puts("  inference : #{verdict(inference)}")
    IO.puts("  loop      : #{verdict(loop)}")
    IO.puts("  latency   : #{verdict(latency)}")
    %{inference: inference, loop: loop, latency: latency}
  end

  @doc """
  Check 1: real ONNX inference on the device CPU.

  Feeds a fixed synthetic observation (`[1, 2, 3]`) straight into the policy so
  the assertion is deterministic and independent of the live (initially zero)
  robot state — we are testing that `Ortex.run/2` runs on ARM and returns the
  exact trained mapping, not the robot.
  """
  def check_inference do
    with {:ok, state} <- ONNX.init(policy_opts()) do
      obs = %{input: Nx.tensor([1.0, 2.0, 3.0], type: :f32)}
      {%{action: action}, _state} = ONNX.act(obs, state)
      got = action |> Nx.backend_transfer() |> Nx.to_flat_list()

      if close?(got, @expected) do
        {:ok, %{got: got}}
      else
        {:error, {:wrong_output, got, @expected}}
      end
    end
  rescue
    e -> {:error, {:exception, Exception.message(e)}}
  end

  @doc "Check 2: the full policy→actuator loop against the simulated robot."
  def check_loop do
    :ok = ensure_armed()

    result =
      BB.Policy.run(@robot, ONNX, %{task: :bench},
        policy_opts: policy_opts(),
        rate_hz: 20,
        timeout: 1_000
      )

    # The linear policy never signals :done, so a clean :timeout means the loop
    # ran (observed, inferred, commanded the sim actuators) for the whole window.
    case result do
      {:ok, :timeout} -> {:ok, %{ran: true}}
      {:ok, :completed} -> {:ok, %{ran: true}}
      other -> {:error, other}
    end
  rescue
    e -> {:error, {:exception, Exception.message(e)}}
  end

  @doc "Check 3: inference latency distribution at a target rate."
  def check_latency(samples, rate_hz) do
    {:ok, state} = ONNX.init(policy_opts())
    obs = %{input: Nx.tensor([0.1, 0.2, 0.3], type: :f32)}

    times =
      for _ <- 1..samples do
        {us, {_action, _st}} = :timer.tc(fn -> ONNX.act(obs, state) end)
        us
      end

    sorted = Enum.sort(times)
    budget_us = div(1_000_000, rate_hz)
    p50 = percentile(sorted, 50)
    p99 = percentile(sorted, 99)

    IO.puts("  latency over #{samples} runs (budget #{budget_us} µs @ #{rate_hz} Hz):")
    IO.puts("    p50=#{p50} µs  p99=#{p99} µs  max=#{List.last(sorted)} µs")

    if p99 <= budget_us do
      {:ok, %{p50: p50, p99: p99, budget_us: budget_us}}
    else
      {:warn, %{p50: p50, p99: p99, budget_us: budget_us, note: "p99 over budget"}}
    end
  rescue
    e -> {:error, {:exception, Exception.message(e)}}
  end

  # --- helpers -------------------------------------------------------------

  defp ensure_armed do
    if BB.Safety.armed?(@robot) do
      :ok
    else
      {:ok, cmd} = @robot.arm(%{})
      # arm completes as {:ok, :armed} or {:ok, :armed, [next_state: :idle]}.
      case BB.Command.await(cmd, 5_000) do
        {:ok, :armed} -> :ok
        {:ok, :armed, _opts} -> :ok
        other -> {:error, {:arm_failed, other}}
      end
    end
  end

  defp close?(got, expected) do
    Enum.zip(got, expected) |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-4 end)
  end

  defp percentile(sorted, p) do
    idx = max(0, round(p / 100 * length(sorted)) - 1)
    Enum.at(sorted, idx)
  end

  defp verdict({:ok, info}), do: "PASS #{inspect(info)}"
  defp verdict({:warn, info}), do: "WARN #{inspect(info)}"
  defp verdict({:error, reason}), do: "FAIL #{inspect(reason)}"
end
