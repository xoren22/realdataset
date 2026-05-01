#!/usr/bin/env python
"""Export interpretable tree graphs for fixed-support RF feature sets."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text

from experiments.baselines import load_samples, rmse
from experiments.sampling_assisted_r101 import SEED, SPARSE_COLS, SUPPORT_COUNT, add_sparse_features, support_table
from src.paths import REPO_ROOT, require_data


FIGS = REPO_ROOT / "figs"
RUNS = REPO_ROOT / "runs"
MODELS = REPO_ROOT / "models"


def render_dot(dot_path: Path) -> None:
    for ext in ("png", "svg"):
        out = dot_path.with_suffix(f".{ext}")
        subprocess.run(["dot", f"-T{ext}", str(dot_path), "-o", str(out)], check=True)


def export_tree_graph(tree, dot_path: Path, title: str, feature_names: list[str], max_depth: int) -> None:
    export_graphviz(
        tree,
        out_file=str(dot_path),
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        precision=2,
        max_depth=max_depth,
        label="root",
    )
    text = dot_path.read_text()
    text = text.replace("Tree {", f'Tree {{\nlabel="{title}";\nlabelloc=t;\nfontsize=20;')
    dot_path.write_text(text)
    render_dot(dot_path)


def export_rf_visualization(
    data: pd.DataFrame,
    feature_cols: list[str],
    prefix: str,
    title_name: str,
) -> None:
    train = data[data["is_support"] == 0].copy()
    forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(train[feature_cols], train["rssi"])
    forest_pred = forest.predict(train[feature_cols])

    actual_tree = forest.estimators_[0]
    export_tree_graph(
        actual_tree,
        FIGS / f"{prefix}_actual_tree0_depth4.dot",
        f"{title_name} actual forest tree 0, truncated to depth 4",
        feature_names=feature_cols,
        max_depth=4,
    )
    actual_text = export_text(actual_tree, feature_names=feature_cols, max_depth=4, decimals=2)
    (RUNS / f"{prefix}_actual_tree0_depth4.txt").write_text(actual_text)

    surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=SEED)
    surrogate.fit(train[feature_cols], forest_pred)
    surrogate_pred = surrogate.predict(train[feature_cols])
    export_tree_graph(
        surrogate,
        FIGS / f"{prefix}_surrogate_depth3.dot",
        f"shallow surrogate tree mimicking {title_name} predictions",
        feature_names=feature_cols,
        max_depth=3,
    )
    surrogate_text = export_text(surrogate, feature_names=feature_cols, max_depth=3, decimals=2)
    (RUNS / f"{prefix}_surrogate_depth3.txt").write_text(surrogate_text)

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": forest.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(RUNS / f"{prefix}_full_fixed_support_importance.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "artifact": "actual_tree0",
                "train_query_n": int(len(train)),
                "target": "true RSSI",
                "depth": int(actual_tree.get_depth()),
                "leaves": int(actual_tree.get_n_leaves()),
                "forest_train_rmse": rmse(train["rssi"].to_numpy(), forest_pred),
                "forest_train_r2": float(r2_score(train["rssi"], forest_pred)),
            },
            {
                "artifact": "surrogate_depth3",
                "train_query_n": int(len(train)),
                "target": f"{title_name} prediction",
                "depth": int(surrogate.get_depth()),
                "leaves": int(surrogate.get_n_leaves()),
                "surrogate_vs_forest_rmse": rmse(forest_pred, surrogate_pred),
                "surrogate_vs_forest_r2": float(r2_score(forest_pred, surrogate_pred)),
            },
        ]
    )
    summary.to_csv(RUNS / f"{prefix}_tree_visualization_summary.csv", index=False)
    joblib.dump(forest, MODELS / f"{prefix}_full_fixed_support_unscaled.joblib")


def main() -> int:
    require_data()
    FIGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    samples_path = RUNS / "sampling_assisted_r101_samples.csv"
    if samples_path.exists():
        r101_data = pd.read_csv(samples_path)
        r101_data = r101_data[(r101_data["fold"] == 0) & (r101_data["rt_mode"] == "zero_rt")].copy()
        data = r101_data
    else:
        samples = load_samples()
        supports = support_table(samples, SUPPORT_COUNT)
        data = add_sparse_features(samples, supports)

    export_rf_visualization(
        data=data,
        feature_cols=SPARSE_COLS,
        prefix="sparse_rf",
        title_name="sparse_rf",
    )
    if "r101_pl_pred" in data.columns:
        export_rf_visualization(
            data=data,
            feature_cols=SPARSE_COLS + ["r101_pl_pred"],
            prefix="r101_pl_rf",
            title_name="r101_pl_rf",
        )

    all_paths = [
        FIGS / "sparse_rf_actual_tree0_depth4.png",
        FIGS / "sparse_rf_actual_tree0_depth4.svg",
        RUNS / "sparse_rf_actual_tree0_depth4.txt",
        FIGS / "sparse_rf_surrogate_depth3.png",
        FIGS / "sparse_rf_surrogate_depth3.svg",
        RUNS / "sparse_rf_surrogate_depth3.txt",
        RUNS / "sparse_rf_full_fixed_support_importance.csv",
        RUNS / "sparse_rf_tree_visualization_summary.csv",
        MODELS / "sparse_rf_full_fixed_support_unscaled.joblib",
        FIGS / "r101_pl_rf_actual_tree0_depth4.png",
        FIGS / "r101_pl_rf_actual_tree0_depth4.svg",
        RUNS / "r101_pl_rf_actual_tree0_depth4.txt",
        FIGS / "r101_pl_rf_surrogate_depth3.png",
        FIGS / "r101_pl_rf_surrogate_depth3.svg",
        RUNS / "r101_pl_rf_surrogate_depth3.txt",
        RUNS / "r101_pl_rf_full_fixed_support_importance.csv",
        RUNS / "r101_pl_rf_tree_visualization_summary.csv",
        MODELS / "r101_pl_rf_full_fixed_support_unscaled.joblib",
    ]
    existing_paths = [path for path in all_paths if path.exists()]
    manifest = pd.DataFrame({"path": [str(path) for path in existing_paths]})
    manifest.to_csv(RUNS / "rf_tree_visualization_manifest.csv", index=False)

    print("Wrote:")
    for path in existing_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
