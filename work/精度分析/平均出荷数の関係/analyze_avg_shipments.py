#!/usr/bin/env python3

"""
Generate analytics about the relationship between average shipments per product
and Simple Error for the 2025-01 to 2025-03 period.

This script reads the prepared accuracy summary, computes descriptive statistics,
produces a scatter plot with a simple regression trend, and writes a Markdown
report summarising the findings. All outputs are saved alongside this script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    correlation: float
    slope: float
    intercept: float
    r_squared: float
    avg_ship_summary: pd.Series
    simple_error_summary: pd.Series
    quartile_table: pd.DataFrame
    top_avg_shipments: pd.DataFrame
    bottom_avg_shipments: pd.DataFrame
    shipment_pareto: pd.DataFrame
    top_20_products: pd.DataFrame
    top_50_products: pd.DataFrame


def describe_correlation(value: float) -> str:
    """Return a qualitative Japanese description of the correlation."""
    if math.isnan(value):
        return "相関を特定できず"

    abs_val = abs(value)
    if abs_val < 0.1:
        return "ほぼ相関なし"
    if abs_val < 0.3:
        strength = "弱い"
    elif abs_val < 0.5:
        strength = "中程度の"
    else:
        strength = "強い"

    direction = "正" if value > 0 else "負"
    return f"{strength}{direction}相関傾向"


def describe_r_squared(value: float) -> str:
    """Return qualitative description for R^2."""
    if math.isnan(value):
        return "説明力は不明"
    if value < 0.1:
        return "説明力は非常に低い"
    if value < 0.3:
        return "説明力は限定的"
    if value < 0.5:
        return "一部説明可能"
    return "高い説明力"


def configure_japanese_font() -> str | None:
    """Set a Japanese-capable font if available to avoid missing glyphs."""
    candidate_paths = [
        Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
        Path("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"),
        Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
        Path("/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf"),
        Path("/usr/share/fonts/opentype/ipafont-mincho/ipamp.ttf"),
    ]

    # include system fonts discovered dynamically as fallback
    candidate_paths.extend(Path(p) for p in font_manager.findSystemFonts())

    for font_path in candidate_paths:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        except Exception:
            continue

        if not font_name:
            continue

        plt.rcParams["font.family"] = [font_name]
        current = plt.rcParams.get("font.sans-serif", [])
        if isinstance(current, str):
            current = [current]
        plt.rcParams["font.sans-serif"] = [font_name] + [f for f in current if f != font_name]
        return font_name

    return None


JAPANESE_FONT = configure_japanese_font()

def apply_font_settings() -> None:
    if not JAPANESE_FONT:
        return
    plt.rcParams["font.family"] = [JAPANESE_FONT]
    current = plt.rcParams.get("font.sans-serif", [])
    if isinstance(current, str):
        current = [current]
    plt.rcParams["font.sans-serif"] = [JAPANESE_FONT] + [f for f in current if f != JAPANESE_FONT]


def load_product_name_map(input_dir: Path) -> Dict[str, str]:
    """Load material_key → product_name mapping from parquet/Excel sources."""
    mapping: Dict[str, str] = {}

    parquet_candidates = [
        input_dir / "df_confirmed_order_input_yamasa.parquet",
        input_dir / "df_confirmed_order_input_yamasa_fill_zero.parquet",
    ]

    for path in parquet_candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["material_key", "product_name"])
        df = df.dropna(subset=["material_key", "product_name"])
        df["material_key"] = df["material_key"].astype(str).str.strip()
        df["product_name"] = df["product_name"].astype(str).str.strip()
        for row in df.drop_duplicates("material_key").itertuples(index=False):
            mapping.setdefault(row.material_key, row.product_name)

    # Excel fallback in case some codes are missing from parquet
    excel_candidates = sorted(input_dir.glob("*.xlsx"))
    for excel_path in excel_candidates:
        try:
            workbook = pd.ExcelFile(excel_path)
        except Exception:
            continue
        for sheet in workbook.sheet_names:
            try:
                sheet_df = workbook.parse(sheet, usecols=["品番", "品名"])
            except Exception:
                continue
            sheet_df = sheet_df.dropna(subset=["品番", "品名"])
            sheet_df["品番"] = sheet_df["品番"].astype(str).str.strip()
            sheet_df["品名"] = sheet_df["品名"].astype(str).str.strip()
            for code, name in sheet_df.drop_duplicates("品番").itertuples(index=False):
                mapping.setdefault(code, name)

    return mapping


def load_data(base_dir: Path) -> pd.DataFrame:
    """Load accuracy summary data and compute average shipments per product."""
    summary_path = (
        base_dir.parent.parent
        / "report_2025-01_to_03_imp2"
        / "summary"
        / "accuracy_summary.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV at {summary_path}")

    df = pd.read_csv(summary_path)

    if "actual_sum" not in df or "n_samples" not in df or "simple_error" not in df:
        raise ValueError(
            "accuracy_summary.csv must contain 'actual_sum', 'n_samples', and 'simple_error'"
        )

    project_root = base_dir.parent.parent.parent
    input_dir = project_root / "input"
    product_map = load_product_name_map(input_dir)

    df = df.copy()
    df["avg_shipment"] = df["actual_sum"] / df["n_samples"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["avg_shipment", "simple_error"])
    df["product_name"] = df["material_key"].map(product_map)
    return df


def analyse(df: pd.DataFrame) -> AnalysisResult:
    """Compute key statistics describing the relationship."""
    x = df["avg_shipment"]
    y = df["simple_error"]

    correlation = x.corr(y)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    avg_ship_summary = x.describe(percentiles=[0.25, 0.5, 0.75])
    simple_error_summary = y.describe(percentiles=[0.25, 0.5, 0.75])

    quartiles = pd.qcut(x, 4, labels=False, duplicates="drop")
    quartile_table = (
        df.assign(quartile=quartiles)
        .groupby("quartile", dropna=True)
        .agg(
            avg_shipment_mean=("avg_shipment", "mean"),
            simple_error_mean=("simple_error", "mean"),
            simple_error_median=("simple_error", "median"),
            count=("material_key", "count"),
            avg_shipment_min=("avg_shipment", "min"),
            avg_shipment_max=("avg_shipment", "max"),
        )
        .reset_index()
    )

    top_avg_shipments = df.sort_values("avg_shipment", ascending=False).head(10)
    bottom_avg_shipments = df.sort_values("avg_shipment", ascending=True).head(10)

    shipment_order = df.sort_values("actual_sum", ascending=False).reset_index(drop=True)
    total_shipment = shipment_order["actual_sum"].sum()
    if total_shipment <= 0:
        shipment_order["shipment_share"] = 0.0
        shipment_order["cumulative_share"] = 0.0
    else:
        shipment_order["shipment_share"] = shipment_order["actual_sum"] / total_shipment
        shipment_order["cumulative_share"] = shipment_order["shipment_share"].cumsum()

    def capture_threshold(threshold: float) -> pd.DataFrame:
        if shipment_order.empty:
            return shipment_order.copy()
        mask = shipment_order["cumulative_share"] <= threshold
        subset = shipment_order.loc[mask].copy()
        if subset.empty:
            subset = shipment_order.head(1).copy()
        elif subset["cumulative_share"].max() < threshold and len(subset) < len(shipment_order):
            subset = shipment_order.loc[: subset.index[-1] + 1].copy()
        subset["cumulative_share_pct"] = subset["cumulative_share"] * 100
        subset["shipment_share_pct"] = subset["shipment_share"] * 100
        return subset

    top_20_products = capture_threshold(0.20)
    top_50_products = capture_threshold(0.50)
    shipment_order["cumulative_share_pct"] = shipment_order["cumulative_share"] * 100
    shipment_order["shipment_share_pct"] = shipment_order["shipment_share"] * 100

    return AnalysisResult(
        correlation=correlation,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        avg_ship_summary=avg_ship_summary,
        simple_error_summary=simple_error_summary,
        quartile_table=quartile_table,
        top_avg_shipments=top_avg_shipments,
        bottom_avg_shipments=bottom_avg_shipments,
        shipment_pareto=shipment_order,
        top_20_products=top_20_products,
        top_50_products=top_50_products,
    )


def format_interval(row: pd.Series) -> str:
    """Format the average shipment range for display."""
    left = row["avg_shipment_min"]
    right = row["avg_shipment_max"]
    return f"{left:,.0f} - {right:,.0f}"


def build_table(quartile_table: pd.DataFrame) -> str:
    """Convert the quartile summary into a Markdown table."""
    rows: Iterable[str] = [
        "|平均出荷数レンジ|件数|平均Simple Error|中央値Simple Error|",
        "|---|---|---|---|",
    ]
    for _, row in quartile_table.iterrows():
        band = format_interval(row)
        rows.append(
            "|{band}|{count}|{mean:.2f}|{median:.2f}|".format(
                band=band,
                count=int(row["count"]),
                mean=row["simple_error_mean"],
                median=row["simple_error_median"],
            )
        )
    return "\n".join(rows)


def build_ranking_table(rank_df: pd.DataFrame) -> str:
    """Create Markdown table for ranking data."""
    rows: Iterable[str] = [
        "|順位|SKU|商品名|平均出荷数|Simple Error|",
        "|---|---|---|---|---|",
    ]
    for idx, (_, row) in enumerate(rank_df.iterrows(), start=1):
        product_name = ""
        if isinstance(row.get("product_name"), str) and row["product_name"].strip():
            product_name = row["product_name"].replace("|", "／")
        else:
            product_name = "-"
        rows.append(
            "|{rank}|{sku}|{name}|{avg:.1f}|{err:.2f}|".format(
                rank=idx,
                sku=row["material_key"],
                name=product_name,
                avg=row["avg_shipment"],
                err=row["simple_error"],
            )
        )
    return "\n".join(rows)


def build_share_table(share_df: pd.DataFrame) -> str:
    """Create Markdown table summarising shipment share contributions."""
    rows: Iterable[str] = [
        "|順位|SKU|商品名|出荷量合計|構成比(%)|累積構成比(%)|",
        "|---|---|---|---|---|---|",
    ]
    for idx, (_, row) in enumerate(share_df.iterrows(), start=1):
        name = "-"
        if isinstance(row.get("product_name"), str) and row["product_name"].strip():
            name = row["product_name"].replace("|", "／")
        rows.append(
            "|{rank}|{sku}|{name}|{shipment:,.0f}|{share:.2f}|{cum:.2f}|".format(
                rank=idx,
                sku=row["material_key"],
                name=name,
                shipment=row["actual_sum"],
                share=row.get("shipment_share_pct", 0.0),
                cum=row.get("cumulative_share_pct", 0.0),
            )
        )
    return "\n".join(rows)


def plot_relationship(df: pd.DataFrame, result: AnalysisResult, output_path: Path) -> None:
    """Create scatter plot with regression line."""
    plt.style.use("seaborn-v0_8-whitegrid")
    apply_font_settings()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["avg_shipment"], df["simple_error"], alpha=0.6, edgecolor="k", linewidth=0.3)

    x_vals = np.linspace(df["avg_shipment"].min(), df["avg_shipment"].max(), 200)
    y_vals = result.intercept + result.slope * x_vals
    ax.plot(x_vals, y_vals, color="tab:red", linewidth=2, label="Linear Trend")

    ax.set_title("Average Shipment vs Simple Error")
    ax.set_xlabel("Average Shipment (Actual)")
    ax.set_ylabel("Simple Error")
    ax.legend()
    ax.margins(x=0.05, y=0.1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rankings(result: AnalysisResult, output_path: Path) -> None:
    """Plot top and bottom 10 average shipment rankings."""
    plt.style.use("seaborn-v0_8-whitegrid")
    apply_font_settings()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=False)

    top = result.top_avg_shipments.copy()
    bottom = result.bottom_avg_shipments.copy()

    def compose_display_name(row: pd.Series) -> str:
        name = row.get("product_name")
        if isinstance(name, str) and name.strip():
            return f"{name} ({row['material_key']})"
        return row["material_key"]

    top["display_name"] = top.apply(compose_display_name, axis=1)
    bottom["display_name"] = bottom.apply(compose_display_name, axis=1)

    top = top.iloc[::-1]  # reverse for horizontal bar order
    bottom = bottom.iloc[::-1]

    bar_bottom = axes[0].barh(
        bottom["display_name"], bottom["avg_shipment"], color="tab:blue", alpha=0.7
    )
    axes[0].set_title("Lowest Avg Shipments (Top 10)")
    axes[0].set_xlabel("Average Shipment")
    axes[0].set_ylabel("Product (Name / SKU)")

    bar_top = axes[1].barh(
        top["display_name"], top["avg_shipment"], color="tab:green", alpha=0.7
    )
    axes[1].set_title("Highest Avg Shipments (Top 10)")
    axes[1].set_xlabel("Average Shipment")
    axes[1].set_ylabel("Product (Name / SKU)")

    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    bottom_max = bottom["avg_shipment"].max() if not bottom.empty else 0
    top_max = top["avg_shipment"].max() if not top.empty else 0

    if bottom_max > 0:
        axes[0].set_xlim(0, bottom_max * 1.6)
    if top_max > 0:
        axes[1].set_xlim(0, top_max * 1.1)

    def annotate_bars(bars, values, axis, ref):
        if not values.size:
            return
        offset = ref * 0.05 if ref > 0 else 0.2
        for bar, value in zip(bars, values):
            width = bar.get_width()
            axis.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"SE {value:.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
            )

    annotate_bars(bar_bottom, bottom["simple_error"].values, axes[0], bottom_max)
    annotate_bars(bar_top, top["simple_error"].values, axes[1], top_max)

    fig.suptitle("Average Shipment Rankings (Jan-Mar 2025)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_shipment_pareto(result: AnalysisResult, output_path: Path) -> None:
    """Plot cumulative shipment share (Pareto chart)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    apply_font_settings()

    pareto_df = result.shipment_pareto.head(20).copy()
    fig, ax1 = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(pareto_df))
    bars = ax1.bar(
        indices,
        pareto_df["actual_sum"],
        color="tab:purple",
        alpha=0.7,
        label="Shipment Volume",
    )
    ax1.set_xlabel("Top Products by Shipment")
    ax1.set_ylabel("Shipment Volume")
    ax1.set_xticks(indices)
    tick_labels = pareto_df.apply(
        lambda row: f"{row['product_name']} ({row['material_key']})"
        if isinstance(row.get("product_name"), str) and row["product_name"].strip()
        else row["material_key"],
        axis=1,
    )
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        indices,
        pareto_df["cumulative_share_pct"],
        color="tab:red",
        marker="o",
        linewidth=2,
        label="Cumulative Share (%)",
    )
    ax2.set_ylabel("Cumulative Share (%)")
    ax2.set_ylim(0, 105)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    simple_error = pareto_df["simple_error"]
    ax3.plot(
        indices,
        simple_error,
        color="tab:orange",
        marker="s",
        linewidth=2,
        label="Average Simple Error",
    )
    ax3.set_ylabel("Average Simple Error")

    for idx, (bar, cum) in enumerate(zip(bars, pareto_df["cumulative_share_pct"])):
        ax2.text(
            idx,
            min(cum + 3, 102),
            f"{cum:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:red",
        )

    err_max = simple_error.max() if not simple_error.empty else 0
    offset = err_max * 0.05 if err_max > 0 else 0.5
    for idx, err in enumerate(simple_error):
        ax3.text(
            idx,
            err + offset,
            f"SE {err:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:orange",
        )

    fig.suptitle("Shipment Share Pareto (Top 20 SKUs)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    lines_labels = [ax.get_legend_handles_labels() for ax in (ax1, ax2, ax3)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.01, 0.98))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(df: pd.DataFrame, result: AnalysisResult, report_path: Path) -> None:
    """Write Markdown report summarising findings."""
    corr_desc = describe_correlation(result.correlation)
    r2_desc = describe_r_squared(result.r_squared)
    impact_per_100 = result.slope * 100
    if abs(impact_per_100) < 0.01:
        slope_clause = "変化はほぼ見られない"
    else:
        direction_word = "増加" if impact_per_100 > 0 else "減少"
        slope_clause = f"{abs(impact_per_100):.2f} ポイント{direction_word}"

    low_band = result.quartile_table.iloc[0]
    high_band = result.quartile_table.iloc[-1]
    worst_band = result.quartile_table.sort_values(
        "simple_error_mean", ascending=False
    ).iloc[0]

    high_mean = high_band["simple_error_mean"]
    low_mean = low_band["simple_error_mean"]
    if abs(high_mean - low_mean) > 1:
        if high_mean > low_mean:
            quartile_comment = (
                "- 出荷上位帯でSimple Errorがやや高く、低出荷帯では誤差が抑えられている。"
            )
        else:
            quartile_comment = (
                "- 低出荷帯でSimple Errorが高く、出荷ボリュームが増えるほど誤差が低下する傾向。"
            )
    else:
        quartile_comment = "- 四分位ごとのSimple Error平均に大きな差異は見られない。"

    if impact_per_100 > 0.05:
        trend_bullet = "- 出荷ボリュームが大きいSKUほどSimple Errorがわずかに上昇する傾向。"
    elif impact_per_100 < -0.05:
        trend_bullet = "- 出荷ボリュームが大きいSKUほどSimple Errorがわずかに低下する傾向。"
    else:
        trend_bullet = "- 出荷ボリュームによるSimple Errorの大きな差は見られない。"

    worst_quartile = int(worst_band["quartile"])
    if worst_quartile == 0:
        focus_segment = "低出荷帯"
        noise_comment = (
            "3. 出荷ボリュームが小さいSKUではデータ不足によるノイズが考えられるため、類似商品のデータ併用も検討。"
        )
    elif worst_quartile == result.quartile_table["quartile"].max():
        focus_segment = "高出荷帯"
        noise_comment = (
            "3. 高出荷帯のSKUは欠品・販促等のイベント影響を把握し、需要急変への対応策を準備。"
        )
    else:
        focus_segment = "中位出荷帯"
        noise_comment = (
            "3. 中位出荷帯のSKUでは構成比の変化や販路別需要など追加特徴量の検討が有効。"
        )

    lines = []
    lines.append("# 平均出荷数とSimple Errorの関係分析")
    lines.append("")
    lines.append("## 1. 概況")
    lines.append(
        f"- 対象データ: {len(df):,} SKU (2025年1月〜3月期間の予測結果サマリー)"
    )
    lines.append(
        "- 平均出荷数は `actual_sum / n_samples` として算出し、Simple Errorは提供された値を利用"
    )
    lines.append("")
    lines.append("## 2. 相関と回帰")
    lines.append(
        f"- ピアソン相関係数: {result.correlation:.3f} ({corr_desc})"
    )
    lines.append(
        f"- 単回帰の傾き: {result.slope:.4f} (平均出荷数が100増えるとSimple Errorは約 {slope_clause})"
    )
    lines.append(f"- 決定係数 R^2: {result.r_squared:.3f} ({r2_desc})")
    lines.append("")
    lines.append("## 3. 指標の分布")
    lines.append("- 平均出荷数 (Actual平均) の要約統計値:")
    lines.append(
        f"  - 中央値: {result.avg_ship_summary['50%']:.1f}, 四分位範囲: "
        f"{result.avg_ship_summary['25%']:.1f} - {result.avg_ship_summary['75%']:.1f}"
    )
    lines.append("- Simple Error の要約統計値:")
    lines.append(
        f"  - 中央値: {result.simple_error_summary['50%']:.2f}, 四分位範囲: "
        f"{result.simple_error_summary['25%']:.2f} - {result.simple_error_summary['75%']:.2f}"
    )
    lines.append("")
    lines.append("## 4. 平均出荷数帯別のSimple Error")
    lines.append(build_table(result.quartile_table))
    lines.append("")
    lines.append("主な傾向:")
    lines.append(trend_bullet)
    lines.append(quartile_comment)
    lines.append(
        "- Simple Errorの高いSKUについては時系列の需要変動やイベント影響を確認し、異常要因の切り分けを行う。"
    )
    lines.append("")
    lines.append("## 5. 平均出荷数ランキング")
    lines.append("### 低出荷SKU (平均出荷数が少ない順)")
    lines.append(build_ranking_table(result.bottom_avg_shipments))
    lines.append("")
    lines.append("### 高出荷SKU (平均出荷数が多い順)")
    lines.append(build_ranking_table(result.top_avg_shipments))
    lines.append("")
    total_shipment = df["actual_sum"].sum()
    top20_share_pct = (
        result.top_20_products["actual_sum"].sum() / total_shipment * 100
        if total_shipment > 0
        else float("nan")
    )
    top50_share_pct = (
        result.top_50_products["actual_sum"].sum() / total_shipment * 100
        if total_shipment > 0
        else float("nan")
    )
    lines.append("## 6. 出荷量シェア分析")
    if not math.isnan(top20_share_pct):
        lines.append(
            f"- 全SKU出荷量合計: {total_shipment:,.0f}、そのうち上位シェア20%カバーSKUは合計 {top20_share_pct:.2f}% を占有"
        )
    else:
        lines.append("- 出荷量合計が0のためシェアを算出できませんでした")
    if not math.isnan(top50_share_pct):
        lines.append(f"- 上位シェア50%カバーSKUは合計 {top50_share_pct:.2f}% を占有")
    lines.append("")
    lines.append("### 20%シェアを構成するSKU")
    lines.append(build_share_table(result.top_20_products))
    lines.append("")
    lines.append("### 50%シェアを構成するSKU")
    lines.append(build_share_table(result.top_50_products))
    lines.append("")
    lines.append("## 7. 推奨アクション")
    lines.append(
        f"1. {focus_segment} (平均出荷数レンジ: {format_interval(worst_band)}) のSKUを優先的に深掘り。"
    )
    lines.append(
        "2. Simple Errorが10を超えるSKUの誤差要因を時系列グラフで可視化し、パターンを特定。"
    )
    lines.append(noise_comment)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_chart = base_dir / "average_shipment_vs_simple_error.png"
    ranking_chart = base_dir / "average_shipment_rankings.png"
    pareto_chart = base_dir / "shipment_pareto.png"
    report_path = base_dir / "analysis_report.md"

    df = load_data(base_dir)
    result = analyse(df)
    plot_relationship(df, result, output_chart)
    plot_rankings(result, ranking_chart)
    plot_shipment_pareto(result, pareto_chart)
    write_report(df, result, report_path)


if __name__ == "__main__":
    main()
