<!--
SPDX-FileCopyrightText: 2026 James Harton

SPDX-License-Identifier: Apache-2.0
-->

# BB.Policy Usage Rules

`bb_policy` runs **learned policies** — neural networks (typically exported to
ONNX) that map observations to actions — on a [Beam Bots](https://hexdocs.pm/bb)
robot, with full safety integration. For BB framework basics, see `bb`'s rules
(`mix usage_rules.sync <file> bb:all`); this file covers only what's specific to
policies.

## Core principles

1. **A policy is a `BB.Policy` behaviour module, not the thing you declare.**
   The behaviour is `init/1 → reset/1 → observe/3 → act/2 →
   action_to_commands/3`. You rarely write one: `BB.Policy.ONNX` is the ready
   implementation — you pass it a model and specs. What you declare on the robot
   is one of the three *runners* below, with `policy: BB.Policy.ONNX`.
2. **Pick the runner by lifetime.** A policy runs as a bounded episode
   (`BB.Policy.Command`, `BB.Policy.Runner`) or a standing behaviour
   (`BB.Policy.Controller`). `{:done, state}` from `act/2` *completes* an
   episode but only *resets* a standing controller — "done" has no terminal
   meaning there.
3. **Safety is not bypassed.** Every runner checks `BB.Safety.armed?/1` before
   applying a command; a mid-episode disarm halts the episode. The robot must be
   armed (through the command system, never by calling `BB.Safety` directly) or
   the policy drives nothing.
4. **Ortex is an optional dependency you must add yourself.** `BB.Policy.ONNX`
   guards on it at runtime; `init/1` returns `{:error, :ortex_not_available}`
   when it's absent.

## Loading a model

`BB.Policy.ONNX` takes these `policy_opts` (see `BB.Policy.ONNX` for the full
spec):

* `:model` (required) — a path string, or `{:priv, app, "models/x.onnx"}` to
  resolve against a packaged app's priv dir at runtime (use this on Nerves).
* `:observation` (required) — ordered list of `{source, joints}` where `source`
  is `:positions` or `:velocities`, e.g. `[positions: [:hip, :knee]]`. Builds
  the model input vector.
* `:action` (required) — ordered list of `{joints, kind}` where `kind` is
  `:position`, `:velocity`, or `:effort`, e.g. `[{[:hip, :knee], :effort}]`.
  Maps output columns to actuator commands.
* `:normalizer` — a `BB.Policy.Normalizer` or path to its JSON. Optional; every
  feature key the specs reference must have stats or `init/1` fails with
  `{:error, {:missing_normalizer_stats, …}}`.

## Wiring it into a robot

Standing behaviour — declare a `controller`:

```elixir
controllers do
  controller :balance,
    {BB.Policy.Controller,
     policy: BB.Policy.ONNX,
     policy_opts: [
       model: "priv/models/balance.onnx",
       observation: [positions: [:hip, :knee], velocities: [:hip, :knee]],
       action: [{[:hip, :knee], :effort}]
     ],
     rate: 50},
    simulation: :start
end
```

Bounded, awaitable, reactor-usable — declare a `command`:

```elixir
commands do
  command :pick_mug do
    handler {BB.Policy.Command,
      policy: BB.Policy.ONNX,
      policy_opts: [
        model: "priv/models/pick_mug.onnx",
        observation: [positions: [:shoulder, :elbow, :wrist]],
        action: [{[:shoulder, :elbow, :wrist], :position}]
      ],
      rate_hz: 20}
    allowed_states [:idle]
    timeout :timer.seconds(30)
  end
end
```

One-shot, imperatively — `BB.Policy.run/4` (blocks until the episode ends):

```elixir
{:ok, :completed} =
  BB.Policy.run(MyRobot.Robot, BB.Policy.ONNX, %{task: :pick_mug},
    policy_opts: [
      model: "priv/models/pick_mug.onnx",
      observation: [positions: [:shoulder, :elbow, :wrist]],
      action: [{[:shoulder, :elbow, :wrist], :position}]
    ],
    rate_hz: 20,
    timeout: :timer.seconds(30)
  )
```

`mix igniter.install bb_policy` scaffolds the `command` form with `TODO`
placeholders for the model path and joints.

## Anti-patterns

- **Don't expect a policy to move a disarmed robot.** A `BB.Policy.Controller`
  silently idles while disarmed and a `BB.Policy.Command`/`Runner` episode ends
  as `:disarmed`. Arm the robot through the command system first.
- **Don't declare a controller and expect it to run in simulation.** `controller`
  defaults to `simulation: :omit`, so a policy controller does *nothing* under
  `:kinematic`/`:mock` sim unless you set `simulation: :start`.
- **Don't forget `{:ortex, "~> 0.1"}` in your deps.** It is optional here and not
  pulled in transitively; without it `BB.Policy.ONNX.init/1` errors out.
- **Don't use `observation_keys:`/`action_keys:`.** The real ONNX options are the
  ordered `:observation`/`:action` specs above.

## Further reading

- [bb_policy docs](https://hexdocs.pm/bb_policy)
- `bb`'s safety rules (`bb:safety-and-commands`) and
  [Understanding Safety](https://hexdocs.pm/bb/understanding-safety.html)
