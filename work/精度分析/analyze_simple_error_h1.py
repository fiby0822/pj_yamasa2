#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR
REPORT_PATH = BASE_DIR / "simple_error_summary_h1.txt"
FONT_PATH = Path("/home/ubuntu/yamasa2/work/script/tmp/fonts/NotoSansCJKjp-Regular.otf")

PREDICTION_FILES: List[Tuple[str, Path]] = [
    ("2025年1-3月", Path("/home/ubuntu/yamasa2/work/data/predictions/product_level_predictions_20251022_054623.parquet")),
    ("2025年4-6月", Path("/home/ubuntu/yamasa2/work/data/predictions/product_level_predictions_20251022_055004.parquet")),
]

AGGREGATED_INPUT = Path("/home/ubuntu/yamasa2/work/data/input/df_confirmed_order_input_yamasa_fill_zero.parquet")


def configure_font() -> None:
    if FONT_PATH.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(FONT_PATH))
        font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False


def build_usage_type_map() -> Dict[str, str]:
    df = pd.read_parquet(AGGREGATED_INPUT, columns=["material_key", "usage_type"])
    df["material_key"] = df["material_key"].astype(str)
    df["usage_type"] = df["usage_type"].astype(str)
    usage_map = (
        df.groupby("material_key")["usage_type"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )
    return usage_map


def load_predictions() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, path in PREDICTION_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        part = pd.read_parquet(path)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["material_key"] = df["material_key"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["material_key", "date"])
    return df


def compute_simple_error(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    denom_pred = df["predicted"].abs()
    simple_error = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mask = denom_pred > 0
    simple_error.loc[valid_mask] = df.loc[valid_mask, "abs_error"] / denom_pred.loc[valid_mask] * 100.0
    simple_error.loc[df["actual"] == 0] = 0.0
    df["simple_error_pct"] = simple_error
    return df


def aggregate_metrics(df: pd.DataFrame, usage_map: Dict[str, str]) -> pd.DataFrame:
    grouped = df.groupby("material_key")
    summary = grouped.agg(
        simple_error_pct=("simple_error_pct", "mean"),
        actual_zero_ratio=("actual", lambda x: (x <= 0).mean()),
        actual_std=("actual", "std"),
        actual_mean=("actual", "mean"),
        actual_sum=("actual", "sum"),
        predicted_sum=("predicted", "sum"),
        n_days=("actual", "count"),
    ).reset_index()
    summary["actual_cv"] = summary["actual_std"] / summary["actual_mean"].replace(0, np.nan)
    summary["usage_type"] = summary["material_key"].map(usage_map).fillna("unknown")
    summary["bias_ratio"] = (summary["predicted_sum"] - summary["actual_sum"]) / summary["actual_sum"].replace(0, np.nan)
    return summary


def plot_histogram_by_usage(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="simple_error_pct",
        hue="usage_type",
        bins=20,
        stat="percent",
        element="step",
        palette="Set2",
    )
    plt.title("Simple Error 分布（利用区分別）")
    plt.xlabel("Simple Error (%)")
    plt.ylabel("構成比 (%)")
    plt.axvline(df["simple_error_pct"].median(), color="gray", linestyle="--", label="全体中央値")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "simple_error_hist_usage.png", bbox_inches="tight")
    plt.close()


def plot_box_by_usage(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="usage_type", y="simple_error_pct")
    plt.title("利用区分別 Simple Error 分布")
    plt.xlabel("利用区分")
    plt.ylabel("Simple Error (%)")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "simple_error_box_usage.png", bbox_inches="tight")
    plt.close()


def plot_scatter(df: pd.DataFrame, x_col: str, x_label: str, filename: str, log_x: bool = False) -> None:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=x_col,
        y="simple_error_pct",
        scatter_kws={"alpha": 0.6, "s": 40},
        line_kws={"color": "black"},
        truncate=False,
    )
    plt.title(f"Simple Error と {x_label} の関係")
    plt.xlabel(x_label)
    plt.ylabel("Simple Error (%)")
    if log_x:
        plt.xscale("log")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, bbox_inches="tight")
    plt.close()


def write_report(df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("2025年1-6月 Simple Error 分析レポート")
    lines.append("")
    lines.append(f"対象SKU数: {len(df)}")
    lines.append(f"Simple Error中央値: {df['simple_error_pct'].median():.2f}% / 平均: {df['simple_error_pct'].mean():.2f}%")
    for usage in ["business", "household"]:
        subset = df[df["usage_type"] == usage]
        if not subset.empty:
            lines.append(
                f"・{usage} SKU: 件数 {len(subset)}, Simple Error中央値 {subset['simple_error_pct'].median():.2f}%, 平均 {subset['simple_error_pct'].mean():.2f}%"
            )
    corr_zero = df[["simple_error_pct", "actual_zero_ratio"]].corr().iloc[0, 1]
    corr_std = df[["simple_error_pct", "actual_std"]].corr().iloc[0, 1]
    corr_cv = df[["simple_error_pct", "actual_cv"]].corr().iloc[0, 1]
    lines.append("")
    lines.append(f"相関係数: ゼロ実績比率 vs Simple Error = {corr_zero:.3f}")
    lines.append(f"相関係数: 実績標準偏差 vs Simple Error = {corr_std:.3f}")
    lines.append(f"相関係数: 実績CV vs Simple Error = {corr_cv:.3f}")

    lines.append("")
    lines.append("観察ポイント")
    lines.append("1. 業務用SKUのSimple Errorは家庭用よりもやや高く、大口顧客特有の変動や補充リードタイムが影響している可能性がある。")
    lines.append("2. ゼロ実績比率が高いSKUほどSimple Errorがわずかに増加しており、間欠需要の対応強化が求められる。")
    lines.append("3. 実績標準偏差が高いSKUではSimple Errorがむしろ低下傾向にある一方、実績CVが高いSKUではSimple Errorが顕著に上昇しており、相対的なばらつきへの対応が必要。")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    configure_font()
    usage_map = build_usage_type_map()

    pred_df = load_predictions()
    pred_df = compute_simple_error(pred_df)

    summary_df = aggregate_metrics(pred_df, usage_map)
    summary_df.to_csv(BASE_DIR / "simple_error_summary_h1.csv", index=False)

    plot_histogram_by_usage(summary_df)
    plot_box_by_usage(summary_df)
    plot_scatter(
        summary_df,
        x_col="actual_zero_ratio",
        x_label="ゼロ実績比率",
        filename="simple_error_vs_zero_ratio.png",
    )
    plot_scatter(
        summary_df,
        x_col="actual_std",
        x_label="実績標準偏差",
        filename="simple_error_vs_actual_std.png",
    )
    plot_scatter(
        summary_df,
        x_col="actual_cv",
        x_label="実績CV",
        filename="simple_error_vs_actual_cv.png",
    )

    write_report(summary_df)


if __name__ == "__main__":
    main()
