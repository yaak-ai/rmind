{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nix-gl-host = {
      url = "github:numtide/nix-gl-host";
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
            gh
            git
            git-lfs
            openssh
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
            MATURIN_NO_INSTALL_RUST = "1";
            PYO3_PYTHON = "${python312}/bin/python";
            UV_PYTHON = "${python312}/bin/python";
            UV_NO_MANAGED_PYTHON = "1";
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
                  uvWrapped = writeShellScriptBin "uv" ''
                    if [[ -e /etc/NIXOS ]]; then
                      export LD_LIBRARY_PATH="${uvLibraryPath}''${NIX_LD_LIBRARY_PATH:+:''${NIX_LD_LIBRARY_PATH}}"
                      exec ${lib.getExe uv} "$@"
                    fi

                    export LD_LIBRARY_PATH="${uvLibraryPath}''${LD_LIBRARY_PATH:+:''${LD_LIBRARY_PATH}}"
                    exec ${lib.getExe nixglhost} ${lib.getExe uv} "$@"
                  '';
                in
                {
                  default = mkShell {
                    packages = commonPackages ++ [
                      nixglhost
                      uvWrapped
                    ];
                    env = commonEnv;
                    shellHook = ''
                      if [[ -e /etc/NIXOS ]]; then
                        export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
                      else
                        export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
                      fi
                    '';
                  };
                };
            }
            .${system};
        };
    };
}
