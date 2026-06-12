<!--
SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>

SPDX-License-Identifier: Apache-2.0
-->

# End-to-end testing on a Raspberry Pi Zero 2 W (Nerves)

This is the plan for Phase 6: proving `bb_policy` end-to-end on real ARM
hardware — **real ONNX inference on-device**, the **full policy→actuator loop**
(in `simulation: :kinematic`, no physical servos needed), and **latency/jitter
measurement**. Target: Raspberry Pi Zero 2 W (aarch64, Cortex-A53, glibc),
Nerves system `rpi0_2`.

> Status: **plan, not yet executed.** Each step lists what could fail and how
> we'll know. Two items are genuinely unproven and must be validated early
> (see "The two real unknowns").

## Why this is now achievable

The feared blocker — "where do you get an aarch64 onnxruntime?" — largely
dissolves:

- Nerves `rpi0_2` runs a **64-bit aarch64 / glibc** userland on OTP 28 (it runs
  the A53 in 64-bit mode for the Erlang JIT). The matching Rust target is
  `aarch64-unknown-linux-gnu`.
- `ortex` → `ort` 2.0-rc.8 → ONNX Runtime 1.19. `ort`'s default
  `download-binaries` feature is **cross-target-aware**: it reads cargo's
  `TARGET` and fetches the binary for *that* triple, not the build host's.
- For `aarch64-unknown-linux-gnu`, pyke ships a **static** `libonnxruntime.a`
  (verified: the `msort_static` artifact is a real AArch64 ELF archive; the
  dynamic variant 404s). So onnxruntime gets **statically linked into the NIF
  `.so`** — there is no separate `libonnxruntime.so` to find, ship, or
  `ORT_DYLIB_PATH`. Self-contained NIF.
- Nerves cross-compiles Rustler NIFs already: it sets `RUSTLER_TARGET` (mapping
  `aarch64-nerves-linux-gnu` → `aarch64-unknown-linux-gnu`) and exports the
  cross toolchain (`CC`, `AR`, `NERVES_SDK_SYSROOT`). Ortex is a stock Rustler
  crate, so this "just works" in principle.

## Proven path: build natively on an aarch64 Linux host ✅

The macOS cross-build blocker below is *entirely* a consequence of building on an
Apple host (the `cc` crate injects `-arch arm64`/`-gfull`). Building on a real
**aarch64 Linux** machine makes the Ortex/`ring` compile **native**, so that
problem disappears. The chosen host is `server-haus.local` (NixOS, aarch64,
glibc 2.40) — same CPU family as the Pi.

> **Confirmed 2026-06-02:** `MIX_TARGET=rpi0_2 ORTEX=1 mix firmware` built
> successfully on `server-haus.local`. Ortex's `ring`/onnxruntime compiled
> cleanly (no `-arch` error), producing
> `deps/ortex/.../aarch64-unknown-linux-gnu/release/libortex.so` →
> `deps/ortex/priv/native/ortex.so`, and a **74.8 MB** `bb_policy_firmware.fw`
> (≈7 MB larger than the no-ortex build — the statically-linked onnxruntime).
> Still pending: flashing it and running `BbPolicyFirmware.Bench.run/0` on the
> actual Pi (NIF load + inference + loop + latency).

Flow: `rsync` the project up (minus `_build`/`deps`), build the firmware there in
`nix develop`, `rsync` the `.fw` back, and flash from the Mac (which has the SD
reader at `/dev/disk5`).

```bash
# from the Mac, repo root:
rsync -az --delete --exclude _build/ --exclude deps/ --exclude '/.git/' \
  --exclude '*.fw' bb_policy/ server-haus.local:~/code/bb_policy/

# on the server, in the dev shell:
ssh server-haus.local
cd ~/code/bb_policy && nix develop
  rustup default stable                       # native target = aarch64-unknown-linux-gnu
  mix archive.install hex nerves_bootstrap     # one-time
  cd examples/bb_policy_firmware
  export MIX_TARGET=rpi0_2 ORTEX=1
  mix deps.get && mix firmware                  # native Ortex build — no -arch bug

# back on the Mac: pull the image and flash
rsync -az server-haus.local:~/code/bb_policy/examples/bb_policy_firmware/_build/rpi0_2_dev/nerves/images/bb_policy_firmware.fw /tmp/
sudo fwup -a -i /tmp/bb_policy_firmware.fw -d /dev/disk5 -t complete
```

Open question still to validate even on a native Linux host: the Nerves build
links against its **own Buildroot sysroot**, not the host's glibc 2.40 — so the
resulting NIF should be ABI-correct for the device. The rootfs *does* ship
`libstdc++.so.6.0.32` (confirmed in the staging sysroot), so the C++ runtime the
NIF needs is present; optionally add `-static-libstdc++` for extra safety.

## Build progress so far (2026-06-02)

A real cross-build was attempted on an aarch64 macOS host. Findings:

- **Firmware without ortex: builds cleanly** (`mix firmware` → 68 MB `.fw`). The
  Nerves toolchain, system, robot DSL, and the rest of the app are all fine on
  `rpi0_2`. Several host build tools had to be added to the flake along the way:
  `pkg-config` (vintage_net_wifi), `squashfsTools` (image), `coreutils-prefixed`
  (`gstat`), `rustup` + `fwup`. Also added: `rel/vm.args.eex` + `rel/env.sh.eex`
  (a hand-built Nerves app lacks the generated release files), and the robot's
  `:disarm` command must use `allowed_states([:idle])` (`:executing` isn't a
  state of this robot).
- **Firmware WITH ortex: blocked at the Rust cross-compile.** Two layers of
  toolchain wiring were solved, then a third blocked it:
  1. Ortex compiles its crate via `Rustler.Compiler.compile_crate/3` and takes
     the cargo target from app config — **fixed** by
     `config :ortex, Ortex.Native, target: "aarch64-unknown-linux-gnu"` in
     `config/rpi0_2.exs` (committed). Without it cargo built for the host.
  2. The cross-std must come from the same toolchain as `cargo` — **fixed** by
     using `rustup` alone (not the nixpkgs `rustc`/`cargo`, which lacked the
     `aarch64-unknown-linux-gnu` `rustlib`). After `nix develop`:
     `rustup default stable && rustup target add aarch64-unknown-linux-gnu`.
  3. **Blocker:** `ort`'s default `download-binaries` feature pulls in `ring`
     (TLS for the HTTPS fetch). `ring`'s `cc`-crate build script injects
     Apple-host flags (`-arch arm64`, `-gfull`) and hands them to the Nerves
     Linux GCC, which rejects them. Target-specific `CC_*`/`CFLAGS_*` env vars
     did not override Nerves' generic `CFLAGS`. This is the known macOS-host →
     linux-gnu `cc`-crate conflict, made worse by Nerves exporting its own
     `CFLAGS`.

### Getting past the `ring` blocker — two paths (both remove `ring`)

`ring` is only present because ort downloads onnxruntime over HTTPS at build
time. Remove the download and `ring` disappears with it:

- **Path 1 — vendor the static lib (recommended).** Pre-download
  `ortrs-msort_static-v1.19.0-aarch64-unknown-linux-gnu.tgz`, and build ort with
  `default-features = false` + `ORT_LIB_LOCATION=/path/to/onnxruntime/lib`
  (system/static linking, no download → no `ring`). The friction: ortex pins
  ort with default features in its own `Cargo.toml`, so this needs a
  `[patch]`/fork of ortex (or a vendored copy with the Cargo feature flipped).
- **Path 2 — build the cross-compile in a Linux environment.** Use `cross` (its
  Docker images bundle a proper `aarch64-unknown-linux-gnu` GCC), or build the
  whole firmware in a Linux container, so the `cc` crate never sees an Apple
  host. Avoids the macOS-specific flag injection entirely. Heavier setup, but
  it's the path with the most prior art for `ring`-on-cross.

The simplest *first green* remains **Option D** (plain Raspberry Pi OS 64-bit +
native build on the Pi): no cross-compile, no `cc`-crate host confusion — ortex
builds natively. Use it to prove the inference + loop + latency story on this
exact A53, then return to Nerves cross-compile as a packaging concern.

## The two real unknowns (validate these first)

1. **Does the `rpi0_2` rootfs ship `libstdc++.so`?** onnxruntime is C++; even
   statically linked, the NIF needs `libstdc++` (+ libm/libpthread) at runtime.
   If it's missing, the NIF won't load. → On-device: `find / -name 'libstdc++*'`.
   If absent, we need a custom Nerves system with `BR2_INSTALL_LIBSTDCPP`.
2. **Does the C++ 19 MB static archive link cleanly through the Nerves GCC?**
   glibc-to-glibc lowers the risk vs musl, but it's unproven for this combo.
   → It either links during `mix firmware` or it doesn't; the error is explicit.

No public evidence exists of Ortex running under Nerves — we'd be first. Budget
time to validate (1) and (2) before anything else, and keep the escape hatch
(plain Raspberry Pi OS, native build) in mind if they bite hard.

## Architecture of the test — already built

`bb_policy` is a **library**, not an app — Nerves firmware lives in a separate
consuming project. That harness already exists under **`examples/bb_policy_firmware/`** (a
self-contained Nerves project; it is excluded from the published hex tarball):

```
examples/bb_policy_firmware/
├── mix.exs                 # nerves + {:bb, ...} + {:bb_policy, path: "../.."}; rpi0_2 system
├── config/{config,host,rpi0_2}.exs
├── lib/bb_policy_firmware/
│   ├── application.ex      # starts the robot in simulation: :kinematic
│   ├── robot.ex            # minimal 3-joint robot (matches the test model's I/O)
│   └── bench.ex            # the E2E + latency harness — BbPolicyFirmware.Bench.run/0
└── priv/models/linear_policy.onnx   # the test model (copied from test/fixtures)
```

The robot runs in `simulation: :kinematic`, so `BB.Sim.Actuator` stands in for
real servos: it accepts position commands and publishes `BeginMotion`, and the
open-loop position estimator feeds position back — closing the policy→actuator
loop without hardware.

> **Already verified on the host** (x86/Mac, real Ortex, Elixir 1.19/OTP 28):
> the robot compiles + starts in kinematic sim + arms; ONNX inference returns the
> exact `[4.5, 6.5]`; `BB.Policy.run/4` drives the full loop; latency p50≈16 µs /
> p99≈86 µs (trivially under the 50 ms / 20 Hz budget on x86). The Pi will be
> slower but the only *new* thing on-device is the ARM cross-compile of the NIF —
> the application logic is proven.

## Runbook — what you run

Everything below runs from `examples/bb_policy_firmware/`. The library and the harness logic
are already verified on the host; these steps are the ARM build + on-device run.

### 0. Build-host prerequisites

The repo flake now provides `rustup` and `fwup` alongside Elixir 1.19 / OTP 28.
From the repo root, `nix develop`, then:

```bash
rustup target add aarch64-unknown-linux-gnu     # Nerves' Rustler cross-target
mix archive.install hex nerves_bootstrap        # one-time, if not already installed
```

You also need: an SSH public key in `~/.ssh` (the `rpi0_2.exs` config refuses to
build without one — it's how `mix upload` authenticates), and outbound HTTPS to
`parcel.pyke.io` from the build host (ort fetches the aarch64 onnxruntime static
lib at `cargo build` time). A sandboxed/offline build needs Option B.

### 1. Sanity-check on the host first (no Pi needed)

```bash
cd examples/bb_policy_firmware
export ORTEX=1
mix deps.get
iex -S mix              # MIX_TARGET defaults to :host
iex> BbPolicyFirmware.Bench.run()
```

This should print `inference: PASS`, `loop: PASS`, and a latency line — exactly
what was verified during development. If the host run is green, the only
remaining variable is the ARM cross-build.

### 2. Build firmware for the Pi Zero 2 W

```bash
cd examples/bb_policy_firmware
export MIX_TARGET=rpi0_2
export ORTEX=1
mix deps.get
mix firmware            # ← cross-compiles the Ortex NIF + statically links onnxruntime
```

**This is the make-or-break step.** Watch for:
- the `cargo build` for `ortex` fetching the `aarch64-unknown-linux-gnu`
  onnxruntime (network), then
- the C++ static archive linking through the Nerves toolchain (unknown #2).
If it errors here, capture the full output — it's the linker, and it's
diagnosable. Fall back to Option B/D if needed.

### 3. Burn and boot

```bash
mix burn                # first time, to an SD card
# later updates over the network:
# mix upload bb_policy_firmware.local
```

### 4. On-device checks — validate the two unknowns, then the bench

Connect over SSH (`ssh bb_policy_firmware.local`) to reach the IEx console:

```elixir
# Unknown #1 — is libstdc++ present? (NIF won't load without it)
"find / -name 'libstdc++*' 2>/dev/null" |> String.to_charlist() |> :os.cmd() |> IO.puts()

# Unknown #2 was answered by step 2 (it linked) — now confirm the NIF loads + runs:
Code.ensure_loaded?(Ortex)      # => true means the NIF loaded on ARM
BbPolicyFirmware.Bench.run()    # => inference / loop / latency verdicts
```

The latency line is the new on-device datum: p50/p99 on the A53 at 20 Hz. For
the tiny linear test model it should pass comfortably; a real ACT model is where
the p99 number actually matters (R3 — measure worst case, not mean).

## Fallback options (if the two unknowns bite)

- **Option B — vendor the static lib:** download
  `ortrs-msort_static-v1.19.0-aarch64-unknown-linux-gnu.tgz` and point `ort` at
  it (`ORT_LIB_LOCATION`, `default-features = false`). Removes the build-time
  network fetch. Needs an ortex Cargo-feature override.
- **Option C — custom Nerves system:** fork `nerves_system_rpi0_2` to ensure
  `libstdc++` and any flags. High effort; only if (1) fails and can't be overlaid.
- **Option D — escape hatch:** run plain Raspberry Pi OS (64-bit) + Elixir and
  build ortex natively on-device (no cross-compile, normal glibc+libstdc++).
  Proves the ARM inference path with the least risk, at the cost of Nerves'
  immutable-firmware model. Good for a *first* green if Nerves linking stalls.

## What "done" looks like

- [ ] `mix firmware` cross-compiles the Ortex NIF for `rpi0_2` without error.
- [ ] On device: `libstdc++.so` present; the NIF loads.
- [ ] `Ortex.run/2` produces the exact expected output on ARM.
- [ ] The policy→actuator loop completes against the simulated robot.
- [ ] p50/p99 inference latency recorded; verdict on whether 20 Hz holds.

## Repo changes this implies

- Add `rustup` (with the aarch64 target) and `fwup` to `flake.nix` for a
  reproducible cross-build shell.
- The firmware app is a **separate project** (it must not be committed inside
  the `bb_policy` library, which has no `application` mod). It can live beside
  `bb_policy` and depend on it via `path:`.
- Optionally add a `documentation/how-to/` entry (this file) and a make/mix
  alias to drive the build.

## ✅ On-device result (2026-06-02)

Verified on a real Raspberry Pi Zero 2 W (Nerves `rpi0_2`), firmware built
natively on `server-haus.local` and delivered via `mix upload` / `upload.sh`:

- **NIF loads on ARM:** `Ortex.Model` reference live; the aarch64 NIF with
  statically-embedded onnxruntime loads against the Nerves rootfs (glibc +
  libstdc++ 6.0.32) — no ABI issue.
- **Inference correct:** `observe [1,2,3] → act → [4.5, 6.5]` (exact).
- **Latency on the A53** (200 runs of `act/2`, single linear model):
  **p50 = 291 µs, p99 = 932 µs, max = 1184 µs** — vs the 50 ms / 20 Hz budget,
  i.e. p99 is ~54× under budget. Ample headroom for a real ACT model.

Believed to be the first working Ortex-on-Nerves deployment.

### Notes / loose ends
- Ortex 0.1.10 has an `Inspect` bug for `%Ortex.Model{}` (pretty-printing the
  struct raises `Inspect.Algebra` `FunctionClauseError`). Cosmetic only —
  inference is unaffected. Worth an upstream issue.
- A/B validation: a freshly-flashed firmware must be validated
  (`Nerves.Runtime.validate_firmware/0`, or startup-guard auto-validation) or
  the next `mix upload` reverts on reboot. This bit us — the bench-fixed build
  uploaded but reverted to the prior (validated) partition.

### Final result: committed firmware, full bench green (2026-06-02)

Reflashed with the bench-fixed build (`motion-garlic`) so `BbPolicyFirmware.Bench.run/0`
runs the committed code directly (no manual paste). All three checks PASS,
across 7 consecutive runs (200 `act/2` calls each):

- **inference:** `{got: [4.5, 6.5]}` every run.
- **loop:** `{ran: true}` — the full `BB.Policy.Runner` → `BB.Sim.Actuator`
  cycle (arm → observe → infer → command), on-device, with the safety gate.
- **latency** (under realistic load — the loop GenServer runs concurrently):
  p50 ≈ 474–623 µs, p99 ≈ 1.36–1.91 ms, max ≈ 2.0–4.1 ms. vs the 50 ms / 20 Hz
  budget, worst-case is ~12× under.

Note the ~2× higher latency vs the isolated hand-run (p50 291 / p99 932 µs):
the bench measures inference while the live control loop and full supervision
tree are running, so this is latency *under load*. The p50→max spread (~8×) is
the BEAM-scheduler/NIF jitter (R3) — within budget here, but it's the figure
that matters for a heavier (real ACT) model. These numbers are for the 207-byte
8-parameter linear *fixture*, not a real policy — they prove the pipeline and
the overhead floor, not that ACT runs at 20 Hz (that needs a real exported
model, R2).

### Re-validated after the upstream review fixes (2026-06-10)

Re-built and re-flashed (`nephew-dry`, UUID `9e415cee`) from the consolidated
`policy-stack` branch — i.e. *with* the PR-review changes: the `BB.Policy.Effect`
protocol, per-feature normaliser, the in-`Step` safety re-check, and the new
`{:priv, app, relative}` runtime model-path resolution. Two build-environment
changes were needed and are now committed:

- **Build in the OTP-28 `firmware` devShell.** The project moved to Erlang 29 /
  Elixir 1.20, but no Nerves system ships OTP 29 yet (`nerves_system_rpi0_2`
  2.0.x is OTP 28.5), and Nerves requires host==target OTP major. `flake.nix`
  now has `devShells.firmware` (OTP 28.5 / Elixir 1.19): build with
  `nix develop .#firmware` (the library still builds on OTP 29 by default).
- **The example robot now declares a `BB.Policy.Controller`** (`simulation:
  :start`) so the DSL controller path runs on-device, not just the imperative
  `BB.Policy.run/4` bench. Its model uses `{:priv, :bb_policy_firmware,
  "models/linear_policy.onnx"}` — a bare path would freeze the build host's at
  compile time (Spark DSL opts are compile-time literals) and crash-loop the
  controller on the device.

On-device results (USB-gadget link, `bb_policy_firmware.local`):

- **NIF loads on ARM:** `Code.ensure_loaded?(Ortex) == true`.
- **Bench:** inference `{got: [4.5, 6.5]}`, loop `{ran: true}`, latency
  p50 = 587 µs / p99 = 2484 µs / max = 3195 µs (≈20× under the 50 ms budget).
- **DSL controller runs:** the supervised `BB.Controller.Server` for the `:policy`
  controller is alive with `callback_module: BB.Policy.Controller`, the
  `{:priv, …}` model resolved against the device app_dir, and its step counter
  advances ~19–20/s while armed — i.e. a standing `BB.Policy.ONNX` policy driving
  the simulated actuators at 20 Hz, on real hardware, without crash-looping. This
  confirms the runtime model-path fix and the controller addition end-to-end.
