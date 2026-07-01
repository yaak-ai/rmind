{
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
    nix-gl-host = {
      url = "github:numtide/nix-gl-host?shallow=1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, nix-gl-host, ... }:
    {
      devShells.x86_64-linux =
        let
          pkgs = import nixpkgs {
            system = "x86_64-linux";
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [ (final: prev: { cudaPackages = prev.cudaPackages_12_9; }) ];
          };

          nixglhost = nix-gl-host.packages.x86_64-linux.default;
          nvtopWrapped = pkgs.writeShellScriptBin "nvtop" ''
            exec ${nixglhost}/bin/nixglhost ${pkgs.nvtopPackages.nvidia}/bin/nvtop "$@"
          '';
        in
        {
          default =
            let
              uvWrapped = pkgs.writeShellScriptBin "uv" ''
                export LD_LIBRARY_PATH="${
                  pkgs.lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.ffmpeg-headless
                    pkgs.cudaPackages.libnpp
                  ]
                }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                exec ${nixglhost}/bin/nixglhost ${pkgs.uv}/bin/uv "$@"
              '';
            in
            pkgs.mkShell {
              packages = [
                # cuda/gpu stuff
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.nccl
                pkgs.cudaPackages.libcusparse_lt
                pkgs.cudaPackages.libnvshmem
                nixglhost
                nvtopWrapped

                # dev
                pkgs.git
                pkgs.git-lfs
                pkgs.python312
                pkgs.ytt
                pkgs.just
                pkgs.jqp
                pkgs.prek
                uvWrapped

                # misc
                pkgs.nix-output-monitor
              ];

              shellHook = ''
                export UV_PYTHON="${pkgs.python312}/bin/python3"
                export UV_NO_MANAGED_PYTHON="1"
                export UV_PYTHON_DOWNLOADS="never"
                export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
              '';
            };

          ci =
            let
              pkgs = import nixpkgs { system = "x86_64-linux"; };
              uvWrapped = pkgs.writeShellScriptBin "uv" ''
                export LD_LIBRARY_PATH="${
                  pkgs.lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.ffmpeg-headless
                  ]
                }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                exec ${pkgs.uv}/bin/uv "$@"
              '';
            in
            pkgs.mkShell {
              packages = [
                pkgs.python312
                pkgs.ytt
                pkgs.just
                uvWrapped
              ];

              shellHook = ''
                export UV_PYTHON="${pkgs.python312}/bin/python3"
                export UV_NO_MANAGED_PYTHON="1"
                export UV_PYTHON_DOWNLOADS="never"
              '';
            };
        };

      devShells.aarch64-linux.default =
        let
          pkgs = import nixpkgs { system = "aarch64-linux"; };
          uvWrapped = pkgs.writeShellScriptBin "uv" ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.ffmpeg-headless
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec ${pkgs.uv}/bin/uv "$@"
          '';
        in
        pkgs.mkShell {
          packages = [
            pkgs.git
            pkgs.git-lfs
            pkgs.python312
            pkgs.ytt
            pkgs.just
            pkgs.jqp
            pkgs.prek
            uvWrapped
          ];

          shellHook = ''
            export UV_PYTHON="${pkgs.python312}/bin/python3"
            export UV_NO_MANAGED_PYTHON="1"
            export UV_PYTHON_DOWNLOADS="never"
          '';
        };

      devShells.aarch64-darwin.default =
        let
          pkgs = import nixpkgs { system = "aarch64-darwin"; };
          uvWrapped = pkgs.writeShellScriptBin "uv" ''
            export DYLD_FALLBACK_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.ffmpeg-headless
              ]
            }''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
            exec ${pkgs.uv}/bin/uv "$@"
          '';
        in
        pkgs.mkShell {
          packages = [
            pkgs.git
            pkgs.git-lfs
            pkgs.python312
            pkgs.ytt
            pkgs.just
            pkgs.jqp
            pkgs.prek
            uvWrapped
          ];

          shellHook = ''
            export UV_PYTHON="${pkgs.python312}/bin/python3"
            export UV_NO_MANAGED_PYTHON="1"
            export UV_PYTHON_DOWNLOADS="never"
          '';
        };
    };
}
