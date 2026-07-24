<!--
SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>

SPDX-License-Identifier: Apache-2.0
-->

# bb_policy — Project Plan

Roadmap and design decisions for `bb_policy`, the learned-policy package for
Beam Bots. The authoritative requirements are the accepted proposal
[`0002-bb-policy.md`](https://github.com/beam-bots/proposals/blob/main/accepted/0002-bb-policy.md);
this document records *how* we build it, the decisions where we diverge from the
proposal, and the risks surfaced by an ecosystem + ML-stack review.

## 1. Where this sits

`bb_policy` is a **satellite** of `bb` core (currently `0.20.x`, Elixir `~> 1.19`,
OTP 28). It depends on the public APIs of core and adds nothing to core itself.
It is the runtime that executes policies trained elsewhere — training, datasets,
teleop, and vision live in their own packages.

```
                bb (core: DSL, Robot, Runtime, Safety, Actuator, PubSub, Telemetry)
                          ▲
        ┌─────────────────┼─────────────────┐
   bb_pid_controller   bb_policy   bb_ik_fabrik   …   (satellites)
        (this package) ─┘
                          │ optional
                        ortex → onnxruntime
```

## 2. Key decisions

### D1 — Public entry point is `BB.Policy.run/4`; `BB.Motion.run_policy/4` is a deferred core PR

The proposal shows `BB.Motion.run_policy/4`. `BB.Motion` lives in **core** and
exposes no extension/registration hook for satellites, so a satellite cannot add
a function to it. We ship the public API as **`BB.Policy.run/4`** (a thin
delegate to `BB.Policy.Runner.run/4`). Adding the `BB.Motion.run_policy/4`
convenience delegate is tracked as a **separate PR to `bb` core** and is not a
blocker for this package.

### D2 — Runner is a standalone `GenServer` first; `BB.Controller` integration is additive later

The proposal models the runner as a plain `GenServer` with `run/4`. The
ecosystem idiom for *long-lived, declaratively-configured* loops is
`use BB.Controller` (as `BB.PID.Controller` does), which buys supervision,
runtime parameters, and a safety `disarm/1` callback for free.

We do **both, in order**: ship the standalone `GenServer` + `run/4` first
(matches the proposal and the episodic "run a task to completion" use case),
then add a `BB.Controller`-based path for policies that should live in the robot
DSL as a continuously-running controller. The control-loop body (observe → act →
commands → safety → apply) is shared between them.

### D3 — Inference via `Ortex.run/2`, not `Nx.Serving` batched execution

A single robot at a fixed rate issues one inference at a time. `Nx.Serving`'s
batched execution exists to amortise overhead across *concurrent* requests; here
it never fills a batch and its `batch_timeout` (default 100 ms) only adds
latency. Call `Ortex.run/2` directly. Batched serving is reserved for a future
multi-camera / multi-policy scenario.

### D4 — Normalisation is owned by the runtime, not the model

LeRobot (and similar) strip input/output normalisation out of the exported ONNX
graph and ship the dataset statistics separately. `BB.Policy.Normalizer` applies
them at inference time (`:min_max`, `:z_score`, `:identity`). The ONNX file is
**not** assumed to be end-to-end.

### D5 — Target ACT first; first inference on a dev box / simulator

ACT (action-chunking transformer) is small, fast, and the realistic ONNX export
path. Diffusion and VLA (π0) policies have iterative loops that do not export
cleanly today — they are deferred and explicitly out of v1 scope. The first
*real* ONNX inference targets an x86/Mac dev box against a simulated robot; the
Nerves/aarch64 deployment story (the biggest integration risk, see R1) is a
later, separately-scoped phase.

### D6 — `ortex` is an optional dependency

The behaviour, runner, and normaliser have no ML runtime dependency. `ortex` is
`optional: true`; `BB.Policy.ONNX` guards on its presence and degrades to a
clear error when absent, so the package compiles and tests pass without a
compiled onnxruntime.

### D7 — Policies declare message inputs; runtimes collect envelopes

Message inputs are part of a policy's trained observation contract, so an
optional `BB.Policy.inputs/1` callback declares named PubSub paths, per-input
payload types and age limits, an optional alignment tolerance, and a collector
queue threshold after `init/1`. All three
runtimes share an independent latest-envelope collector and pass complete
`BB.Message` values to `observe/3`; they do not strip payloads or query a sample
store on the control-loop hot path. Joint-only policies default to no message
inputs.

After a runtime applies its first effect, a required input becoming missing,
stale, cross-node, or misaligned calls a planned direct
`BB.Safety.emergency_disarm/2` API before a bounded runtime terminates; a
standing controller requests the same intervention and resets. This path must
bypass configurable disarm commands and command capacity. The full contract and
lifecycle are in `docs/design.md`.

## 3. Risks (from the ML-stack review)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **Nerves/aarch64 deployment.** Ortex publishes no precompiled NIFs and pins `ort 2.0-rc`; you must cross-compile the Rustler NIF *and* supply an aarch64 `libonnxruntime` that ort can find. | High | Defer (D5). Spike separately: `load-dynamic` + `ORT_DYLIB_PATH` to a Pi onnxruntime build, or build from source. Treat as its own phase with its own acceptance test. |
| R2 | **LeRobot → ONNX export is not a one-liner.** `select_action` (action queue, ensembling) isn't traceable; export is inference-only subgraphs, often split (vision/transformer), static shapes, opset 11. | High | Owned by a PyTorch-side export pipeline, *out of this repo*. Document the contract (input/output signature + separate stats JSON) the runtime expects. Provide a known-good fixture model. |
| R3 | **NIF blocks the BEAM scheduler.** Ortex inference is a Rustler NIF; a multi-ms call on a normal scheduler hurts soft-real-time jitter. | Medium | Verify Ortex uses a dirty NIF; isolate inference in its own process from the timing loop; cap onnxruntime's intra/inter-op thread pool (leave headroom for BEAM schedulers). Measure **p99**, not mean. |
| R4 | **Silent CPU fallback.** ort silently falls back to CPU if a requested execution provider isn't compiled in. | Low | Log/assert the active EP at load; document that CUDA/CoreML require building Ortex with those features. |
| R5 | **Diffusion/VLA expectations.** Users may expect any LeRobot policy to "just load". | Low | Scope doc + clear error: v1 supports ACT-class static-shape ONNX. |
| R6 | **Policy termination can leave the last actuator command active.** Runner/Command completion, timeout, cancellation, and conversion failure stop without disarming or applying a terminal effect; Controller conversion failure keeps running with its previous effect. Position targets may continue or hold, and velocity/effort commands are persistent. | High | Observation invalidation uses the direct emergency-disarm contract (D7). Issue [#20](https://github.com/beam-bots/bb_policy/issues/20) is the release-blocking safety design for existing termination paths; optional `hold`/`stop` support is not assumed portable. |

## 4. Phased roadmap

Each phase is a vertical slice that leaves the tree green (`mix check`).

### Phase 0 — Scaffold ✅ (this commit)

- Ecosystem-standard project: `mix.exs`, `.check.exs`, `.formatter.exs`,
  `.tool-versions`, `renovate.json`, CI workflow, `LICENSES/` + SPDX headers,
  `AGENTS.md`/`CLAUDE.md`, `CHANGELOG.md`.
- `BB.Policy` behaviour (full callback contract + typedocs).
- Stubs with typed signatures + phase-tagged TODOs: `Runner`, `Normalizer`,
  `ONNX`, `Telemetry`; `BB.Policy.run/4` facade.
- `MockPolicy` test support + behaviour-contract tests.

### Phase 1 — Normalizer (no robot, no ML) ✅

- `:z_score`, `:min_max` (`[0,1]` and `[-1,1]`), `:identity` via `normalize/4` /
  `denormalize/4`, keyed per `:observation`/`:action` space with **per-key**
  strategy; scalar and per-element (tensor) moments; exact round-trip.
- Numerical safety: zero std / `min == max` treated as unit scale (no NaN/Inf).
- `new/1` (+ `new!/1`) validation, `stats_from_samples/3`, and `load/1` parsing
  the exported stats JSON via the stdlib `JSON` module (no extra dependency).
- **Done:** 22 tests + 2 doctests green; `mix format` and `credo --strict` clean.

### Phase 2 — Runner vertical slice (MockPolicy) ✅

- `Runner.init/1` (policy init + reset, deadline, first tick), the
  `handle_info(:tick, …)` loop, and `run/4` (start → monitor → await → teardown).
- Safety gate (`BB.Safety.armed?/1`) checked every tick; a disarm — at start or
  mid-episode — ends the episode with `:disarmed` (intervention, not a retry).
- Deadline/`:timeout`; policy-signalled completion (`act/2` → `{:done, state}`);
  action-conversion errors surfaced as `{:error, {:action_conversion, _}}`.
- Reads robot state via `BB.Robot.Runtime.get_robot_state/1`; applies
  `BB.Policy.ActuatorCommand`s via `BB.Actuator` (new `ActuatorCommand` struct +
  dispatcher, since core has no command type).
- Episode + per-tick inference telemetry (`BB.Policy.Telemetry`).
- **Done:** 14 runner/command tests cover completion, timeout, disarm (both
  forms), init error, conversion error, command application/gating, and
  telemetry. Full suite (38 tests + 2 doctests) green; format, `credo --strict`
  (only phase TODO suggestions), and dialyzer clean.

  The bb boundary (`BB.Safety`, `BB.Robot.Runtime`, `BB.Actuator`) is stubbed
  with Mimic in global mode (the runner runs in its own process). A full
  `simulation: :kinematic` robot integration test is deferred to Phase 3, where
  a real ONNX policy gives it something meaningful to drive.

### Phase 3 — ONNX on dev box (D5) ✅

- `BB.Policy.ONNX` loads a model with `Ortex.load/2`, runs `Ortex.run/2` (direct,
  not batched — D3), wires in `BB.Policy.Normalizer`, and builds
  `BB.Policy.ActuatorCommand`s from the output via a declarative `:observation`
  (`source: joints`) / `:action` (`{joints, kind}`) spec.
- Action-chunking: receding-horizon queue — `act/2` pops one row; refills by
  inferring when the queue empties (a single-action model = infer every tick).
- Optional-dependency hygiene: `init/1` guards with `Code.ensure_loaded?(Ortex)`;
  the two Ortex calls use `apply/3` so the package compiles
  `--warnings-as-errors` and passes dialyzer **without** ortex present.
- Toolchain: `flake.nix` gained `rustc`/`cargo`; `ORTEX=1` builds the NIF and
  `ort`'s `download-binaries` fetches an `aarch64-apple-darwin` onnxruntime.
- Fixture: `test/fixtures/generate_linear.py` builds a tiny static-shape linear
  ONNX (committed as `linear_policy.onnx`); the integration test is tagged
  `:ortex` and auto-excluded when Ortex isn't loaded.
- **Done:** real onnxruntime inference verified end-to-end (obs → normalise →
  `Ortex.run` → denormalise → commands) with exact expected outputs. With
  `ORTEX=1`: 44 tests + 2 doctests green. Without: 38 tests (6 excluded),
  format / warnings-as-errors / `credo --strict` / dialyzer / `reuse lint` clean.

  Not yet done (deferred): driving a live `simulation: :kinematic` robot through
  `BB.Policy.Runner` with this policy (the runner + ONNX are each tested in
  isolation); temporal ensembling; multi-input models (e.g. vision + state).

### Phase 4 — Policy-as-command + reactor integration ✅

- `BB.Policy.Command` is a `use BB.Command` handler (generic, configured by
  `policy:` / `policy_opts:` / `rate_hz:` opts — D-API choice). Declaring it on
  a robot makes a policy a first-class command: awaitable via
  `BB.Command.await/2`, governed by the safety state machine, and usable as a
  `bb_reactor` `command :name` step with no extra glue.
- It runs the policy in the command lifecycle: `init/1` inits the policy;
  `handle_command/3` resets and schedules the first tick; `handle_info(:tick,…)`
  runs one `BB.Policy.Step` and reschedules, stopping `{:ok, :completed}` on
  `{:done, _}` or `{:error, {:action_conversion, _}}` on a conversion failure.
  Timeout is the command's DSL `timeout`; safety disarm uses `BB.Command`'s
  default `handle_safety_state_change/2` (`:disarmed`), which a reactor step
  surfaces as `{:halt, :safety_disarmed}`.
- Refactor: the control cycle (observe → act → action_to_commands → apply) was
  extracted into `BB.Policy.Step`, now shared by `Runner` and `Command` (no
  duplication; the runner keeps scheduling/deadline/owner-reporting).
- **Done:** 8 command tests (init, init-failure, tick loop to completion,
  command application, conversion error, result extraction, disarm handling).
  Full suite green (without Ortex: 46 tests, 6 excluded; with `ORTEX=1`: 52
  tests + 2 doctests); format / warnings-as-errors / credo / dialyzer / reuse
  all clean.

  Not yet done (deferred): an end-to-end test through the real command server +
  a live robot (callbacks are tested directly); the `bb_reactor` workflow
  integration is documented and follows from the standard command contract but
  isn't yet exercised in a test (would need a DSL robot + reactor harness).

### Phase 5 — `BB.Controller` path (D2) + temporal ensembling ✅

- `BB.Policy.Controller` (`use BB.Controller`): runs a policy *continuously* as
  a DSL-declared, supervised controller (vs. the bounded episode of Runner /
  Command). Ticks at `:rate`, runs one `BB.Policy.Step` while armed, idles +
  resets the policy while disarmed, and exposes `disarm/1`. A `{:done, _}` just
  resets and keeps running — a standing controller has no terminal state.
- Temporal ensembling in `BB.Policy.ONNX`: `:temporal_ensemble_coeff` switches
  from the receding-horizon queue to inferring every tick and blending all
  overlapping chunk predictions for the current step with weights
  `exp(-coeff · age)`; stale chunks are pruned. The queue regime stays the
  default.
- **Done:** 8 controller tests (init, init failure, armed step + command, idle
  while disarmed, `:done` keeps running, disarm) + ONNX chunk-queue and
  ensembling tests verifying exact blended values (avg at coeff 0; the
  `exp(-1)`-weighted mix at coeff 1) against a new `chunk_policy.onnx` fixture
  (`[1, 2, 2]` output). With `ORTEX=1`: 61 tests + 2 doctests; without: 52 (9
  excluded). format / warnings-as-errors / credo / dialyzer / reuse all clean.

  Not yet done (deferred): an end-to-end test of the controller inside a live
  supervised robot via the DSL (the callbacks are tested directly).

### Phase 6 — Nerves / aarch64 deployment (R1) ✅ proven on hardware

Target: **Raspberry Pi Zero 2 W** (aarch64 / glibc / OTP 28, Nerves system
`rpi0_2`). R1's crux dissolved: `ort` is cross-target aware and pyke ships a
**static** `aarch64-unknown-linux-gnu` `libonnxruntime.a`, so onnxruntime links
*into* the NIF — no separate `.so` to ship.

- `test_firmware/` — a self-contained Nerves app (excluded from the hex tarball
  via an explicit `package files:` allowlist) depending on `bb_policy` by path.
  A minimal 3-joint robot in `simulation: :kinematic` (so `BB.Sim.Actuator`
  closes the policy→actuator loop with no hardware), plus `BbPolicyFirmware.Bench`
  running three E2E checks: real ONNX inference, the full `BB.Policy.run/4`
  loop, and inference latency p50/p99 vs the 20 Hz budget.
- **The build path that works: native aarch64 Linux.** The macOS cross-build
  fails (the `cc` crate feeds Apple `-arch`/`-gfull` flags to the Nerves GCC via
  `ring`, ort's TLS download dep). Building on a native aarch64 Linux box
  (`server-haus`) makes it a native compile — no Apple flags — and `mix firmware`
  succeeds. flake gained `rustup`/`fwup`/`pkg-config`/`squashfsTools`/
  `coreutils-prefixed`. Full runbook + the toolchain-wiring saga in
  `documentation/how-to/end-to-end-on-pi-zero-2.md`.
- **On-device result (2026-06-02), all three checks PASS, repeatably:** the
  aarch64 NIF (onnxruntime statically embedded) loads against the Nerves rootfs
  (glibc + libstdc++ 6.0.32 — ABI fine); inference `[1,2,3] → [4.5, 6.5]` exact;
  the Runner→Sim.Actuator loop runs with the safety gate; latency under load
  p50 ≈ 0.5 ms, p99 ≈ 1.4–1.9 ms, max ≈ 2–4 ms — ~12× under the 50 ms / 20 Hz
  budget. Believed to be the first working **Ortex-on-Nerves** deployment.

Still open (deferred by design):

- **Real model:** all numbers are for the 207-byte / 8-param linear *fixture* —
  proving the pipeline and overhead floor, not that a real ACT model runs at
  20 Hz. That needs the LeRobot→ONNX export (R2), a Python-side workstream.
- A/B firmware validation: a freshly-flashed image must validate
  (`Nerves.Runtime.validate_firmware/0` / startup-guard) or a later `mix upload`
  reverts on reboot.
- On-target thread-pool tuning + p99 for a real model (R3).
- Core PR for `BB.Motion.run_policy/4` (D1).

### Phase 7 — Message observation inputs (planned, issue #9)

Blocked from release by the existing policy-termination safety work in issue
[#20](https://github.com/beam-bots/bb_policy/issues/20).

- Add optional `BB.Policy.inputs/1` with named exact paths, mandatory per-input
  payload types and age limits, plus optional whole-snapshot alignment and
  collector queue limits.
- Add one shared collector implementation used by Runner, Command, and
  Controller. It runs independently of inference and caches the newest complete
  local `BB.Message` envelope per alias.
- Make `BB.Policy.Step` consume a validated `BB.Policy.ObservationSnapshot`
  carrying messages, collector generation, and earliest expiry; bare external
  fourth-argument maps are no longer an effect-applying path.
- Validate missing, stale, cross-node, and misaligned inputs before inference;
  recheck collector generation/liveness and the snapshot's earliest expiry
  immediately before applying effects.
- Keep joint-only policies compatible through the default empty input spec.
- Treat collection before the first applied effect as warm-up within the
  existing timeout. Afterwards, invalidation calls direct emergency disarm
  before Runner/Command report an error; Controller requests the same disarm
  and resets once.
- Add `BB.Safety.emergency_disarm/2` in core so safety-originated intervention
  cannot be rejected by disarm-command routing or command-category capacity.
  Disarm failure enters `:error` and takes precedence over the observation error.
- Make core register every actuator's required `disarm/1` callback with safety
  before the robot can arm, and treat registration loss while armed as an
  emergency-disarm trigger.
- Add a core arm epoch: actuator command publication stamps the current epoch,
  disarm invalidates it before callbacks run, and actuator servers reject
  missing or stale epochs. This prevents queued commands from applying after a
  disarm callback or later re-arm. The guarantee covers built-in actuator
  effects; custom effects that drive hardware require equivalent epoch and
  safety-registration semantics.
- Extend `BB.Policy.ONNX` with message-derived and multi-input model sources only
  after the common collection path is complete.
- Verify all three lifecycles, collector overload/failure under slow inference,
  and the pre-inference/pre-publication observation gates. Document the remaining
  asynchronous interval before disarm callbacks make hardware safe.

## 5. Acceptance-criteria → phase map

From the proposal's "Acceptance Criteria":

**Must Have** — behaviour (P0) · runner loop at configurable rate (P2) ·
normaliser min-max & z-score (P1) · ONNX via Ortex (P3) · `run_policy` entry
point (P0 as `BB.Policy.run/4`; core delegate P6) · safety integration (P2) ·
timeout handling (P2) · basic telemetry (P2) · ONNX export docs (P3) · tests for
contract/lifecycle/normalisation (P0–P2).

**Should Have** — `BB.Policy.Command` reactor wrapper (P4) · stats from JSON
(P1) · GPU/EP config (P3, dev box) · graceful inference-failure degradation
(P3) · episode reset (P2) · example on simulated robot (P3).

**Won't Have** (separate packages) — native Axon policies, diffusion policies,
training loops, dataset management, vision encoders, Python bridge.

## 6. Proposal questions — current status

1. **Observation source — decided, not implemented.** Robot state remains from
   `BB.Robot.Runtime`; named message inputs use the envelope snapshot contract
   in D7 and `docs/design.md` (P7).
2. **Action representation — implemented.** `BB.Policy.ONNX` uses ordered
   `:action` specifications supporting position, velocity, and effort outputs.
3. **Episode boundaries — partially implemented.** Timeout and
   policy-signalled `:done` are shipped; a public external-cancellation API
   remains open.
4. **Goal specification — open.** Runner and Command retain the goal for
   lifecycle and telemetry, but current policy callbacks do not receive it.
5. **Multi-step actions — implemented.** Receding-horizon queues and temporal
   ensembling both shipped in P5.
6. **Vision input — boundary decided, implementation pending.** Reusable decode,
   geometry, detection, and embedding stages belong to `bb_perception`;
   policy-trained tensor layout, resizing, feature selection, and normalisation
   remain in `observe/3`. Message-derived ONNX inputs follow P7.
7. **Recurrent policies — decided.** Hidden state lives in policy `state`;
   history-window buffering remains the policy's concern.

These are tracked to be closed as the relevant phases land.
