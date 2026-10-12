{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
  };

  outputs = { nixpkgs, ... }: {
    devShells.x86_64-linux.default =
      with import nixpkgs { system = "x86_64-linux"; };
      let
        uvLibraryPath = lib.makeLibraryPath [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
          ffmpeg
        ];
        uvWrapped = writeShellScriptBin "uv" ''
          # driver libs only: nix-ld's dir also ships the system glibc, which clashes with this nixpkgs' glibc
          export LD_LIBRARY_PATH="${uvLibraryPath}:/run/opengl-driver/lib"
          exec ${lib.getExe uv} "$@"
        '';
      in
      mkShell {
        packages = [
          nushell
          gh
          git
          git-lfs
          openssh
          uvWrapped
          ytt
          just
          prek
          skim
          openssl
          ffmpeg
          # private rbyte[yaak] deps (ymmv, m2df) are Rust crates built from git (#261)
          cargo
          rustc
        ];

        env = {
          MATURIN_NO_INSTALL_RUST = "1";
          PYO3_PYTHON = "${python312}/bin/python";
          UV_PYTHON = "${python312}/bin/python";
          UV_NO_MANAGED_PYTHON = "1";
          UV_PYTHON_DOWNLOADS = "never";
          TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib"; # for TorchInductor
        };
      };

    devShells.aarch64-darwin.default =
      with import nixpkgs { system = "aarch64-darwin"; };
      let
        uvLibraryPath = lib.makeLibraryPath [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
          ffmpeg
        ];
        uvWrapped = writeShellScriptBin "uv" ''
          export DYLD_FALLBACK_LIBRARY_PATH="${uvLibraryPath}''${DYLD_FALLBACK_LIBRARY_PATH:+:''${DYLD_FALLBACK_LIBRARY_PATH}}"
          exec ${lib.getExe uv} "$@"
        '';
      in
      mkShell {
        packages = [
          nushell
          gh
          git
          git-lfs
          openssh
          uvWrapped
          ytt
          just
          prek
          skim
          openssl
          ffmpeg
          # private rbyte[yaak] deps (ymmv, m2df) are Rust crates built from git (#261)
          cargo
          rustc
        ];

        env = {
          MATURIN_NO_INSTALL_RUST = "1";
          PYO3_PYTHON = "${python312}/bin/python";
          UV_PYTHON = "${python312}/bin/python";
          UV_NO_MANAGED_PYTHON = "1";
          UV_PYTHON_DOWNLOADS = "never";
        };
      };
  };
}
