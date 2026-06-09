# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import Config

# Cross-compile the Ortex NIF for the Pi Zero 2 W (aarch64 / glibc). Ortex
# compiles its Rust crate via `Rustler.Compiler.compile_crate/3` using the
# target from this app-env key; without it, cargo builds for the host
# (aarch64-apple-darwin) and feeds macOS flags (-arch, -gfull) to the Nerves
# Linux GCC, which fails. The triple must match the Nerves toolchain
# (aarch64-nerves-linux-gnu → Rust aarch64-unknown-linux-gnu).
config :ortex, Ortex.Native, target: "aarch64-unknown-linux-gnu"

# Use Ringlogger as the logger backend and remove :console.
# See https://hexdocs.pm/ring_logger/readme.html for more information on
# configuring ring_logger.

config :logger, backends: [RingLogger]

# Authorize the device to receive firmware using your public key.
# See https://hexdocs.pm/nerves_ssh/readme.html for more information
# on configuring nerves_ssh.

# Extra authorized keys for clients other than the build host — e.g. when
# the firmware is built on a remote aarch64 box (server-haus) but you ssh
# to the device from a different laptop. Drop any *.pub here on the build
# host. See documentation/how-to/end-to-end-on-pi-zero-2.md.
keys =
  ([
     Path.join([System.user_home!(), ".ssh", "id_rsa.pub"]),
     Path.join([System.user_home!(), ".ssh", "id_ecdsa.pub"]),
     Path.join([System.user_home!(), ".ssh", "id_ed25519.pub"])
   ] ++
     Path.wildcard(Path.join([System.user_home!(), ".ssh", "authorized_keys.d", "*.pub"])))
  |> Enum.filter(&File.exists?/1)

if keys == [],
  do:
    Mix.raise("""
    No SSH public keys found in ~/.ssh. An ssh authorized key is needed to
    log into the Nerves device and update firmware on it using ssh.
    See your project's config.exs for this error message.
    """)

config :nerves_ssh,
  authorized_keys: Enum.map(keys, &File.read!/1)

# Configure the network using vintage_net
config :vintage_net,
  regulatory_domain: "00",
  config: [
    {"usb0", %{type: VintageNetDirect}},
    {"eth0",
     %{
       type: VintageNetEthernet,
       ipv4: %{method: :dhcp}
     }},
    {"wlan0", %{type: VintageNetWiFi}}
  ]

config :mdns_lite,
  host: [:hostname, "bb_policy_firmware"],
  ttl: 120,
  services: [
    %{protocol: "ssh", transport: "tcp", port: 22},
    %{protocol: "sftp-ssh", transport: "tcp", port: 22},
    %{protocol: "epmd", transport: "tcp", port: 4369}
  ]
