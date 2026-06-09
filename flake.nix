# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

{
  description = "bb_policy — learned policies for Beam Bots";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    flake-utils.url = "github:numtide/flake-utils";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Pinned to match .tool-versions (Erlang 29 / Elixir 1.20), which stays
        # authoritative for CI. Nix gives a reproducible local shell that agrees.
        erlang = pkgs.beam29Packages;
        elixir = erlang.elixir_1_20;

        # treefmt config — one formatter per language. See formatters.md.
        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true; # Nix (nixfmt-rfc-style)
          programs.mix-format.enable = true; # Elixir (uses .formatter.exs)
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            elixir
            erlang.erlang
            lefthook
            reuse # licence/SPDX lint (mix check runs `reuse lint`)

            # Rust toolchain for building the Ortex NIF (ort 2.0-rc). Only used
            # when ORTEX=1 pulls ortex into the build; ort's download-binaries
            # feature fetches a prebuilt onnxruntime at build time (needs network).
            #
            # Use rustup ALONE (not the nixpkgs rustc/cargo): the Nerves
            # cross-build needs the `aarch64-unknown-linux-gnu` std library, which
            # is installed per-toolchain via `rustup target add`. Mixing the
            # nixpkgs cargo (on PATH) with a rustup-installed target fails with
            # "can't find crate for core" because they don't share rustlib. So we
            # let rustup own both cargo and the cross-std. After `nix develop`:
            #   rustup default stable
            #   rustup target add aarch64-unknown-linux-gnu
            # fwup burns the firmware image.
            rustup
            fwup

            # Host build tools for Nerves C/Rust NIFs. pkg-config is needed by
            # vintage_net_wifi (and friends) to locate libnl in the Nerves
            # sysroot during the cross-build. squashfsTools (mksquashfs) builds
            # the firmware rootfs image; fwup assembles/burns it.
            pkg-config
            squashfsTools

            # Nerves on macOS shells out to GNU coreutils under g-prefixed names
            # (gstat, gfind, gmktemp, …). coreutils-prefixed provides them.
            coreutils-prefixed
          ];

          # Ortex/ort look these up when linking the NIF.
          env = {
            RUSTLER_NIF_VERSION = "2.16";
          };
        };

        # `nix fmt` runs treefmt across the repo.
        formatter = treefmtEval.config.build.wrapper;

        # `nix flake check` verifies everything is formatted.
        checks.formatting = treefmtEval.config.build.check ./.;
      }
    );
}
