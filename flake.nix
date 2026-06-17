{
  description = "Python environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311 # Change this if you need 3.10 or 3.12
            uv
            ruff
            basedpyright
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib 
              pkgs.zlib             
              pkgs.glib             
            ]}"
            
            # Automatically create the virtual env if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Initializing uv virtual environment..."
              uv venv
            fi
            
            # Automatically activate it
            source .venv/bin/activate
          '';
        };
      }
    );
}
