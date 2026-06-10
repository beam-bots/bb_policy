# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BbPolicyFirmware.MixProject do
  use Mix.Project

  @app :bb_policy_firmware
  @all_targets [:rpi0_2]

  def project do
    [
      app: @app,
      version: "0.1.0",
      elixir: "~> 1.19",
      archives: [nerves_bootstrap: "~> 1.13"],
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      releases: [{@app, release()}]
    ]
  end

  # Elixir 1.20 moved preferred targets/envs out of project/0 into this callback.
  def cli do
    [preferred_targets: [run: :host, test: :host]]
  end

  # Run on host with `iex -S mix`, on target with `MIX_TARGET=rpi0_2 mix firmware`.
  def application do
    [
      mod: {BbPolicyFirmware.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  defp deps do
    [
      # Dependencies for all targets
      {:nerves, "~> 1.10", runtime: false},
      {:shoehorn, "~> 0.9.1"},
      {:ring_logger, "~> 0.11.0"},
      {:toolshed, "~> 0.4.0"},

      # The package under test (local) + core.
      {:bb_policy, path: "../.."},
      {:bb, "~> 0.20"},
      # ortex is an OPTIONAL dep of bb_policy, so the consuming app must declare
      # it to pull it into the build. We make it real (non-optional) here, gated
      # by ORTEX=1 — without that env the firmware builds without inference,
      # which is the whole point of this harness, so require it.
      ortex_dep(),

      # Dependencies for host (no target)
      {:nerves_runtime, "~> 0.13.0", targets: @all_targets},
      {:nerves_pack, "~> 0.7.1", targets: @all_targets},

      # Pi Zero 2 W system + toolchain
      {:nerves_system_rpi0_2, "~> 2.0", runtime: false, targets: :rpi0_2}
    ]
  end

  defp release do
    [
      overwrite: true,
      cookie: "#{@app}_cookie",
      include_erts: &Nerves.Release.erts/0,
      steps: [&Nerves.Release.init/1, :assemble],
      strip_beams: Mix.env() == :prod
    ]
  end

  # ortex is bb_policy's optional inference backend; the consuming app must
  # declare it to pull it in. This harness exists to test on-device inference,
  # so building without ORTEX=1 is almost always a mistake — warn loudly.
  defp ortex_dep do
    if System.get_env("ORTEX") in ~w(1 true) do
      {:ortex, "~> 0.1"}
    else
      IO.warn(
        "ORTEX is not set — building bb_policy_firmware WITHOUT the ONNX runtime. " <>
          "BbPolicyFirmware.Bench inference/loop/latency checks will fail. " <>
          "Build with: ORTEX=1 MIX_TARGET=rpi0_2 mix firmware",
        []
      )

      {:ortex, "~> 0.1", only: :__ortex_disabled__}
    end
  end
end
