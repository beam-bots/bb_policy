<!--
SPDX-FileCopyrightText: 2026 James Harton <james@harton.nz>
SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>

SPDX-License-Identifier: Apache-2.0
-->

# BB.Policy

[![CI](https://github.com/beam-bots/bb_policy/actions/workflows/ci.yml/badge.svg)](https://github.com/beam-bots/bb_policy/actions/workflows/ci.yml)
[![Hex version](https://img.shields.io/hexpm/v/bb_policy.svg)](https://hex.pm/packages/bb_policy)
[![Hexdocs](https://img.shields.io/badge/docs-hexdocs-purple)](https://hexdocs.pm/bb_policy)
[![REUSE status](https://api.reuse.software/badge/github.com/beam-bots/bb_policy)](https://api.reuse.software/info/github.com/beam-bots/bb_policy)

Learned policies for [Beam Bots](https://github.com/beam-bots/bb). `bb_policy`
lets robots execute neural-network behaviours that map observations to actions —
train a policy from demonstrations or simulation, export it to ONNX, and deploy
it on real hardware with full safety-system integration.

A policy is a function `π: observation → action`. Given what the robot perceives
(joint positions, velocities, camera frames, forces), the policy outputs what it
should do (target positions, velocities, gripper commands). Inference runs on the
BEAM, in the same runtime as control — so a crashed or slow policy can't take the
robot down with it.

## Status

🚧 **Early scaffold.** The `BB.Policy` behaviour and project conventions are in
place; implementations are landing in phases. See
[`PROJECT_PLAN.md`](https://github.com/beam-bots/bb_policy/blob/main/PROJECT_PLAN.md)
for the roadmap and the design decisions behind it.

## Installation

```elixir
def deps do
  [
    {:bb_policy, "~> 0.1"},
    # ONNX inference is optional — add ortex when you deploy a real model:
    {:ortex, "~> 0.1"}
  ]
end
```

## Usage

```elixir
{:ok, result} =
  BB.Policy.run(MyRobot, BB.Policy.ONNX, %{task: :pick_mug},
    policy_opts: [
      model: "priv/models/pick_mug.onnx",
      normalizer: "priv/models/pick_mug.json",
      observation_keys: [:joint_positions, :joint_velocities, :gripper],
      action_keys: [:target_positions, :target_gripper]
    ],
    rate_hz: 20,
    timeout: :timer.seconds(30)
  )
```

## How it fits the framework

| Concern | Where it lives |
|---------|----------------|
| Map observation → action | `BB.Policy` behaviour |
| Fixed-rate control loop | `BB.Policy.Runner` |
| Input/output scaling | `BB.Policy.Normalizer` |
| ONNX model loading & inference | `BB.Policy.ONNX` (via [Ortex](https://github.com/elixir-nx/ortex)) |
| Safety | `BB.Safety` — the runner only applies commands while armed |
| Observability | `[:bb, :policy, …]` telemetry events |

## Documentation

Generated docs live at <https://hexdocs.pm/bb_policy>. Architecture, decisions,
and the phased roadmap are in
[`PROJECT_PLAN.md`](https://github.com/beam-bots/bb_policy/blob/main/PROJECT_PLAN.md);
contributor conventions are in
[`AGENTS.md`](https://github.com/beam-bots/bb_policy/blob/main/AGENTS.md).

## Licence

Apache-2.0. See [`LICENSES/`](https://github.com/beam-bots/bb_policy/tree/main/LICENSES)
and the SPDX headers on each file ([REUSE](https://reuse.software/)-compliant).
