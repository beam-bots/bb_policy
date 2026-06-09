# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

Application.ensure_all_started(:mimic)

# Boundary modules from bb core that BB.Policy.Runner / ActuatorCommand / ONNX
# call. Tests stub these with Mimic so the runner can be exercised without a
# live robot or hardware.
Mimic.copy(BB.Safety)
Mimic.copy(BB.Robot.Runtime)
Mimic.copy(BB.Robot.State)
Mimic.copy(BB.Actuator)

# Tests tagged :ortex need the Ortex NIF (ORTEX=1 + a Rust toolchain). Skip them
# when it isn't loaded so the suite stays green without onnxruntime.
unless Code.ensure_loaded?(Ortex) do
  ExUnit.configure(exclude: [:ortex])
  IO.puts("ortex not loaded — skipping :ortex-tagged tests")
end

ExUnit.start()
