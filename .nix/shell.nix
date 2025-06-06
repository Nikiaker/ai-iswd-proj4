{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    python3
    graphviz

    python3Packages.pandas
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.scikit-learn
    python3Packages.xgboost
    python3Packages.graphviz
    python3Packages.tabulate
    python3Packages.shap
  ];
}