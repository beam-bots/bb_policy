# SPDX-FileCopyrightText: 2026 James Harton <james@harton.nz>
# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.Policy.MixProject do
  use Mix.Project

  @moduledoc """
  Learned policies for Beam Bots: map observations to actions via ONNX models.
  """

  @version "0.1.0"

  def project do
    [
      aliases: aliases(),
      app: :bb_policy,
      consolidate_protocols: Mix.env() == :prod,
      deps: deps(),
      description: @moduledoc,
      dialyzer: dialyzer(),
      docs: docs(),
      elixir: "~> 1.19",
      elixirc_paths: elixirc_paths(Mix.env()),
      package: package(),
      start_permanent: Mix.env() == :prod,
      version: @version
    ]
  end

  defp dialyzer, do: [plt_add_apps: [:mix]]

  defp package do
    [
      maintainers: [
        "Edgar Gomes de Araujo <talktoedgar@gmail.com>",
        "James Harton <james@harton.nz>"
      ],
      licenses: ["Apache-2.0"],
      # Explicit allowlist so the dev-only Nerves harness in test_firmware/ (and
      # other non-library files) never ships in the published tarball.
      files: ~w(lib documentation .formatter.exs mix.exs README.md CHANGELOG.md LICENSES),
      links: %{
        "Source" => "https://github.com/beam-bots/bb_policy",
        "Sponsor" => "https://github.com/sponsors/jimsynz"
      }
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp docs do
    [
      main: "readme",
      logo: "assets/logo.png",
      extras:
        ["README.md", "CHANGELOG.md"]
        |> Enum.concat(Path.wildcard("documentation/**/*.{md,livemd,cheatmd}")),
      groups_for_extras: [
        Tutorials: ~r/tutorials\//,
        "How-to Guides": ~r/how-to\//,
        Explanation: ~r/topics\//
      ],
      groups_for_modules: [
        Core: [
          BB.Policy
        ],
        Implementations: [
          BB.Policy.ONNX
        ],
        Support: [
          BB.Policy.ActuatorCommand,
          BB.Policy.Normalizer,
          BB.Policy.Step,
          BB.Policy.Telemetry
        ]
      ],
      source_ref: "main",
      source_url: "https://github.com/beam-bots/bb_policy"
    ]
  end

  defp aliases, do: []

  defp deps do
    [
      {:bb, bb_dep("~> 0.20")},
      {:nx, "~> 0.12"},
      ortex_dep(),

      # dev/test
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:ex_check, "~> 0.16", only: [:dev, :test], runtime: false},
      {:ex_doc, ">= 0.0.0", only: [:dev, :test], runtime: false},
      {:git_ops, "~> 2.9", only: [:dev, :test], runtime: false},
      {:igniter, "~> 0.6", only: [:dev, :test], runtime: false},
      {:mimic, "~> 2.2", only: :test, runtime: false},
      {:mix_audit, "~> 2.1", only: [:dev, :test], runtime: false}
    ]
  end

  defp elixirc_paths(env) when env in [:dev, :test], do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp bb_dep(default) do
    case System.get_env("BB_VERSION") do
      nil -> default
      "local" -> [path: "../bb", override: true]
      "main" -> [git: "https://github.com/beam-bots/bb.git", override: true]
      version -> "~> #{version}"
    end
  end

  # ortex compiles a Rust NIF (and downloads an onnxruntime binary), so it needs
  # a Rust toolchain. It is published as an optional dependency — consumers opt
  # in — but is only fetched into *this* repo's build when ORTEX=1, so day-to-day
  # development and CI don't require Rust. BB.Policy.ONNX guards on its presence
  # at runtime via Code.ensure_loaded?/1.
  defp ortex_dep do
    if System.get_env("ORTEX") in ~w(1 true) do
      {:ortex, "~> 0.1", optional: true}
    else
      {:ortex, "~> 0.1", optional: true, only: :__ortex_disabled__}
    end
  end
end
