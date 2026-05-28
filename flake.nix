{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
    flake-utils.url = "github:numtide/flake-utils?shallow=1";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              git
              git-lfs
              uv
              ytt
              just
              jqp
            ]
            ++ lib.optionals stdenv.isDarwin [ ffmpeg ]
            ++ lib.optionals stdenv.isLinux [ stdenv.cc ];

            shellHook = lib.strings.concatLines [
              (lib.optionalString stdenv.isDarwin "export DYLD_FALLBACK_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [ pkgs.ffmpeg ]
              }")
              # torch.compile (inductor) JIT-compiles fused kernels via the
              # system C++ toolchain. Provide nix's gcc/g++ on Linux so the
              # devshell doesn't depend on whatever /usr/bin/{gcc,g++} happens
              # to be (or whether they exist at all).
              (lib.optionalString stdenv.isLinux ''
                export CC=${stdenv.cc}/bin/cc
                export CXX=${stdenv.cc}/bin/c++
              '')
            ];
          };
      }
    );
}
