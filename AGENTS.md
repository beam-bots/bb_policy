<!--
SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>

SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

Guidance for coding assistants working in `bb_policy`.

## Project Overview

`bb_policy` is a satellite package of [Beam Bots](https://github.com/beam-bots/bb).
It runs **learned policies** — neural networks (typically exported to ONNX) that
map observations to actions — on real robots, inside the BEAM, with full safety
integration. It is *not* a training framework, dataset manager, or vision
pipeline; those are separate packages (see "Scope" below and `PROJECT_PLAN.md`).

The authoritative design is the accepted proposal
[`0002-bb-policy.md`](https://github.com/beam-bots/proposals/blob/main/accepted/0002-bb-policy.md).
The phased roadmap and the decisions that diverge from the proposal are in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) — read it before making architectural changes.

## Common Commands

```bash
mix deps.get
mix check --no-retry      # formatter, credo --strict, dialyzer, reuse lint, tests
mix test
mix test test/bb/policy_test.exs:42
mix format
mix credo --strict
mix dialyzer
pipx run reuse lint       # licence/SPDX compliance
```

Develop against a local checkout of core with `BB_VERSION=local` (expects `../bb`),
or the main branch with `BB_VERSION=main`.

`ortex` builds a Rust NIF and downloads an onnxruntime binary, so it is gated
behind `ORTEX=1` (off by default — dev/test/CI don't need Rust). To work on
`BB.Policy.ONNX`, run `ORTEX=1 mix deps.get && ORTEX=1 mix test` inside the dev
shell (which provides `cargo`/`rustc`). The `:ortex`-tagged tests auto-skip when
Ortex isn't loaded. Regenerate the test model with
`test/fixtures/generate_linear.py` (see its header for the nix invocation).

### Nix dev environment

A Nix flake provides a reproducible local toolchain (Erlang 29 / Elixir 1.20,
pinned to match `.tool-versions`, which stays authoritative for CI).

- **Dev shell** — `nix develop`, or let direnv load it on `cd` (`direnv allow`
  once). The shell includes `elixir`, `erlang`, `lefthook`, and `reuse`.
- **Formatting** — `nix fmt` runs treefmt across the repo: `mix format` for
  Elixir (via `.formatter.exs`) and `nixfmt` for the flake. `mix format` and
  `mix check` remain the project-level entry points.
- **Commit gate** — a lefthook `pre-commit` hook formats staged files via
  `nix fmt` and re-stages them. Install with `lefthook install`. This formats
  but does **not** run `credo`/`dialyzer`/`reuse` — run `mix check` for the full
  gate (matching CI).
- New non-Elixir files (`.nix`, `.yml`) still carry SPDX headers so `reuse lint`
  stays green; `flake.lock` is committed.

> Commit messages: do **not** add trailers, attribution, `Co-Authored-By`, or
> `Generated with` footers.

## Architecture

```
BB.Policy            # the behaviour: init/1 reset/1 observe/3 act/2 action_to_commands/3
   ├─ BB.Policy.Step          # one control cycle (observe→act→commands→apply); shared
   ├─ BB.Policy.Runner        # GenServer control loop; BB.Policy.run/4 entry point (bounded episode)
   ├─ BB.Policy.Command       # use BB.Command — run a policy as a robot command (reactor-usable, bounded)
   ├─ BB.Policy.Controller    # use BB.Controller — run a policy continuously (DSL-declared, standing)
   ├─ BB.Policy.ActuatorCommand # command struct a policy emits; dispatched to BB.Actuator
   ├─ BB.Policy.Normalizer    # min-max / z-score / identity scaling (pure Nx)
   ├─ BB.Policy.ONNX          # @behaviour BB.Policy, loads models via Ortex (optional dep)
   └─ BB.Policy.Telemetry     # [:bb, :policy, …] event contract
```

The control cycle (`BB.Policy.Step.run/3`), each tick: `observe/3 → act/2 →
action_to_commands/3 →` apply via `BB.Actuator`. `Runner` and `Command` both
call it; each owns its own scheduling, timeout, safety gate, and telemetry. A
policy reaches its goal by returning `{:done, state}` from `act/2`.

## Key Patterns (match the ecosystem)

- **Namespace:** all modules under `BB.Policy.*`. App is `:bb_policy`; the
  MixProject module is `BB.Policy.MixProject`.
- **Control loop:** schedule ticks with `Process.send_after(self(), :tick, ms)`
  and reschedule from the `handle_info(:tick, …)` body. This mirrors
  `BB.PID.Controller`. Do **not** use `:timer.send_interval` (no backpressure).
- **Safety is not optional.** Check `BB.Safety.armed?/1` before applying any
  command in the loop. A mid-episode disarm halts the episode — it is a safety
  intervention, not a retryable error. Never construct a code path that drives
  actuators while disarmed.
- **Reading robot state:** `BB.Robot.Runtime.get_robot_state/1` then the
  `BB.Robot.State` accessors; or subscribe via `BB.PubSub.subscribe/2,3` and
  cache the latest payloads.
- **Commands out:** a policy's `action_to_commands/3` returns a list of
  `BB.Policy.Effect`s; the control loop applies each via `BB.Policy.Effect.apply/2`
  while armed. `BB.Policy.ActuatorCommand` is the built-in effect — it maps to a
  `BB.Actuator` call (`set_position/4`, `set_velocity/4`, `set_effort/4`,
  `hold/3`, `stop/3`). (Note: this is `bb_policy`'s *own* effect type; core has
  its own command machinery — `BB.Command.*` handlers like `Arm`/`MoveTo` — which
  is a separate concern from a policy's per-tick actuator outputs.)
- **Errors:** prefer structured `BB.Error` types over ad-hoc tuples where they
  fit; all `BB.Error` types must implement `BB.Error.Severity`.
- **Telemetry:** emit through `BB.Telemetry`, event names `[:bb, :policy, …]`.
- **Ortex is an optional dependency.** `BB.Policy.ONNX` must degrade gracefully
  (clear error) when `ortex` is absent — guard with `Code.ensure_loaded?/1`.
- **Inference:** call `Ortex.run/2` directly for the single-robot hot loop. Do
  **not** reach for `Nx.Serving` batched execution — a 20 Hz loop never fills a
  batch and `batch_timeout` only adds latency.
- **Licensing:** every file carries an SPDX header (a copyright line and an
  Apache-2.0 licence identifier); `.md` uses an HTML-comment header. Files that
  can't carry comments get a `<file>.license` sidecar. `mix check` runs
  `reuse lint`.

## Scope (from the proposal)

**This package:** the policy behaviour, runner, normaliser, ONNX implementation,
reactor command wrapper, safety integration, telemetry.

**Not this package** (separate `bb_*` packages): native Axon policies, diffusion
policies, training loops, dataset management, vision encoders, a Python bridge.

## Testing

- `MockPolicy` (`test/support/mock_policy.ex`) is a dependency-free `BB.Policy`
  for exercising the runner and the behaviour contract without Ortex.
- `mimic` is available for mocking; `ortex`-dependent tests must be tagged and
  skippable so CI passes without a compiled onnxruntime.

## Proposals

Feature proposals live in <https://github.com/beam-bots/proposals>.
