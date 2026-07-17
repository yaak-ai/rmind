{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
    flake-parts.url = "github:hercules-ci/flake-parts?shallow=1";
    nix-gl-host = {
      url = "github:numtide/nix-gl-host?shallow=1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        {
          inputs',
          pkgs,
          system,
          ...
        }:
        with pkgs;
        let
          uvLibraryPath = lib.makeLibraryPath [
            stdenv.cc.cc.lib
            zlib
            openssl
            libffi
            ffmpeg
          ];
          commonPackages = [
            nushell
            git
            git-lfs
            ytt
            just
            prek
            skim
            openssl
            ffmpeg
            cargo
            rustc
          ];
          commonEnv = {
            UV_PYTHON = "${python312}/bin/python";
            UV_NO_MANAGED_PYTHON = "1";
            UV_PYTHON_DOWNLOADS = "never";
          };
        in
        {
          devShells =
            {
              aarch64-darwin =
                let
                  uvWrapped = writeShellScriptBin "uv" ''
                    export DYLD_FALLBACK_LIBRARY_PATH="${uvLibraryPath}''${DYLD_FALLBACK_LIBRARY_PATH:+:''${DYLD_FALLBACK_LIBRARY_PATH}}"
                    exec ${lib.getExe uv} "$@"
                  '';
                in
                {
                  default = mkShell {
                    packages = commonPackages ++ [ uvWrapped ];
                    env = commonEnv;
                  };
                };
              x86_64-linux =
                let
                  nixglhost = inputs'.nix-gl-host.packages.default;
                  nixosUv = writeShellScriptBin "uv" ''
                    export LD_LIBRARY_PATH="${uvLibraryPath}''${NIX_LD_LIBRARY_PATH:+:''${NIX_LD_LIBRARY_PATH}}"
                    exec ${lib.getExe uv} "$@"
                  '';
                  ubuntuUv = writeShellScriptBin "uv" ''
                    export LD_LIBRARY_PATH="${uvLibraryPath}''${LD_LIBRARY_PATH:+:''${LD_LIBRARY_PATH}}"
                    exec ${lib.getExe nixglhost} ${lib.getExe uv} "$@"
                  '';
                in
                {
                  nixos = mkShell {
                    packages = commonPackages ++ [ nixosUv ];
                    env = commonEnv // {
                      TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib"; # for TorchInductor
                    };
                  };

                  ubuntu = mkShell {
                    packages = commonPackages ++ [
                      nixglhost
                      ubuntuUv
                    ];
                    env = commonEnv // {
                      TRITON_LIBCUDA_PATH = "/lib/x86_64-linux-gnu"; # for TorchInductor
                    };
                  };
                };
            }
            .${system};
        };
    };
}
