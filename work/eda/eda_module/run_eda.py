#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------------------
# 定数
# --------------------------------------------------------------------------------------
DATA_PATH = Path("/home/ubuntu/yamasa2/work/data/input/df_confirmed_order_input_yamasa_fill_zero.parquet")
EXCEL_INPUT_DIR = Path("/home/ubuntu/yamasa2/input")
FONT_PATH = Path("/home/ubuntu/yamasa2/work/script/tmp/fonts/NotoSansCJKjp-Regular.otf")
FONT_URL = (
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/"
    "NotoSansCJKjp-Regular.otf"
)
GRAPH_DIR = Path("/home/ubuntu/yamasa2/work/eda/graph")
REPORT_DIR = Path("/home/ubuntu/yamasa2/work/eda/report")
USAGE_TYPE_MAP = {
    "business": "業務用",
    "household": "家庭用",
}

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------------------
# データクラス
# --------------------------------------------------------------------------------------
@dataclass
class Insight:
    title: str
    details: str


# --------------------------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------------------------
def ensure_font(font_path: Path = FONT_PATH, font_url: str = FONT_URL) -> str:
    """日本語表示用フォントを設定する。"""
    font_path.parent.mkdir(parents=True, exist_ok=True)
    if not font_path.exists():
        print(f"フォントをダウンロードします: {font_path}")
        urlretrieve(font_url, font_path)
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = font_name
    return font_name


def load_product_name_map(input_dir: Path = EXCEL_INPUT_DIR) -> Dict[str, str]:
    """Excelから品番と品名のマッピングを作成する。"""
    mapping: Dict[str, str] = {}
    if not input_dir.exists():
        return mapping
    for excel_path in sorted(input_dir.glob("*.xlsx")):
        try:
            workbook = pd.ExcelFile(excel_path)
        except Exception as exc:
            print(f"警告: {excel_path} の読込に失敗: {exc}")
            continue
        for sheet in workbook.sheet_names:
            try:
                sheet_df = workbook.parse(sheet, usecols=["品番", "品名"])
            except ValueError:
                continue
            except Exception as exc:
                print(f"警告: {excel_path}::{sheet} の読込に失敗: {exc}")
                continue
            if "品番" not in sheet_df.columns or "品名" not in sheet_df.columns:
                continue
            sheet_df = sheet_df.dropna(subset=["品番", "品名"])
            sheet_df["品番"] = sheet_df["品番"].astype(str).str.strip()
            sheet_df["品名"] = sheet_df["品名"].astype(str).str.strip()
            for row in sheet_df.drop_duplicates("品番").itertuples(index=False):
                mapping.setdefault(row.品番, row.品名)
    return mapping


def prepare_dataframe(df: pd.DataFrame, product_map: Dict[str, str]) -> pd.DataFrame:
    """基本整形と日本語ラベルの付与を行う。"""
    df = df.copy()
    df["product_key"] = df["product_key"].astype(str).str.strip()
    df["product_name"] = (
        df["product_key"].map(product_map).fillna(df["product_name"].astype(str))
    )
    df["store_code"] = df["store_code"].astype(str).str.strip()
    df["usage_type"] = df["usage_type"].astype(str)
    df["usage_type_jp"] = df["usage_type"].map(USAGE_TYPE_MAP).fillna(df["usage_type"])
    df["category_lvl1"] = df["category_lvl1"].astype(str)
    df["category_lvl2"] = df["category_lvl2"].astype(str)
    df["category_lvl3"] = df["category_lvl3"].astype(str)
    df["date"] = pd.to_datetime(df["file_date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    return df


def aggregate_product_daily(df: pd.DataFrame) -> pd.DataFrame:
    """商品別に日次出荷数を集計し、ゼロ出荷間隔分析用の粒度を揃える。"""
    meta_cols = [
        "product_key",
        "product_name",
        "usage_type_jp",
        "usage_type",
        "category_lvl1",
        "category_lvl2",
        "category_lvl3",
    ]
    meta = df[meta_cols].drop_duplicates("product_key")
    daily = (
        df.groupby(["product_key", "date"], as_index=False)["actual_value"]
        .sum()
        .sort_values(["product_key", "date"])
    )
    product_daily = daily.merge(meta, on="product_key", how="left")
    product_daily["month"] = product_daily["date"].dt.to_period("M").astype(str)
    return product_daily


def compute_zero_sequences(product_daily: pd.DataFrame) -> pd.DataFrame:
    """各商品におけるゼロ出荷が連続した期間を抽出する。"""
    sequences: List[Dict[str, object]] = []
    for product_key, group in product_daily.groupby("product_key"):
        sorted_group = group.sort_values("date").reset_index(drop=True)
        zero_flag = sorted_group["actual_value"] == 0
        if zero_flag.sum() == 0:
            continue
        run_id = zero_flag.ne(zero_flag.shift(fill_value=False)).cumsum()
        runs = sorted_group.assign(is_zero=zero_flag, run_id=run_id)
        for _, sub in runs[runs["is_zero"]].groupby("run_id"):
            sequences.append(
                {
                    "product_key": product_key,
                    "product_name": sub["product_name"].iloc[0],
                    "usage_type_jp": sub["usage_type_jp"].iloc[0],
                    "category_lvl1": sub["category_lvl1"].iloc[0],
                    "start_date": sub["date"].iloc[0],
                    "end_date": sub["date"].iloc[-1],
                    "length_days": len(sub),
                }
            )
    if sequences:
        seq_df = pd.DataFrame(sequences)
        seq_df["length_days"] = seq_df["length_days"].astype(int)
        return seq_df
    return pd.DataFrame(
        columns=[
            "product_key",
            "product_name",
            "usage_type_jp",
            "category_lvl1",
            "start_date",
            "end_date",
            "length_days",
        ]
    )


def compute_zero_metrics(
    product_daily: pd.DataFrame, sequences: pd.DataFrame
) -> pd.DataFrame:
    """商品単位のゼロ出荷指標を算出する。"""
    sequence_map: Dict[str, pd.DataFrame] = {}
    if not sequences.empty:
        for key, sub in sequences.groupby("product_key"):
            sequence_map[key] = sub

    metrics: List[Dict[str, object]] = []
    for product_key, group in product_daily.groupby("product_key"):
        sorted_group = group.sort_values("date")
        zero_flag = sorted_group["actual_value"] == 0
        total_days = len(sorted_group)
        zero_days = int(zero_flag.sum())
        zero_share = zero_days / total_days if total_days else np.nan

        positive_values = sorted_group.loc[~zero_flag, "actual_value"]
        mean_positive = float(positive_values.mean()) if not positive_values.empty else 0.0
        std_positive = float(positive_values.std()) if len(positive_values) > 1 else 0.0
        cv_positive = std_positive / mean_positive if mean_positive else np.nan

        positive_dates = sorted_group.loc[~zero_flag, "date"]
        if len(positive_dates) >= 2:
            intervals = positive_dates.diff().dt.days.dropna()
            avg_interval = float(intervals.mean())
            median_interval = float(intervals.median())
            p90_interval = float(intervals.quantile(0.9))
        else:
            avg_interval = np.nan
            median_interval = np.nan
            p90_interval = np.nan

        seq_df = sequence_map.get(product_key)
        if seq_df is not None and not seq_df.empty:
            max_zero = int(seq_df["length_days"].max())
            mean_zero = float(seq_df["length_days"].mean())
            p90_zero = float(seq_df["length_days"].quantile(0.9))
            long_share = float((seq_df["length_days"] >= 30).mean())
        else:
            max_zero = 0
            mean_zero = 0.0
            p90_zero = 0.0
            long_share = 0.0

        metrics.append(
            {
                "product_key": product_key,
                "product_name": sorted_group["product_name"].iloc[0],
                "usage_type_jp": sorted_group["usage_type_jp"].iloc[0],
                "category_lvl1": sorted_group["category_lvl1"].iloc[0],
                "category_lvl2": sorted_group["category_lvl2"].iloc[0],
                "zero_share": zero_share,
                "zero_share_pct": zero_share * 100,
                "zero_days": zero_days,
                "total_days": total_days,
                "avg_zero_streak": mean_zero,
                "max_zero_streak": max_zero,
                "p90_zero_streak": p90_zero,
                "long_zero_share": long_share,
                "mean_positive_demand": mean_positive,
                "cv_positive_demand": cv_positive,
                "avg_positive_interval": avg_interval,
                "median_positive_interval": median_interval,
                "p90_positive_interval": p90_interval,
                "total_volume": float(sorted_group["actual_value"].sum()),
            }
        )

    return pd.DataFrame(metrics)


def summarize_zero_metrics(
    product_daily: pd.DataFrame, metrics: pd.DataFrame
) -> Dict[str, object]:
    """レポート用の集約情報を生成する。"""
    overview = {
        "データ期間": f"{product_daily['date'].min().date()} 〜 {product_daily['date'].max().date()}",
        "分析対象日数": f"{product_daily['date'].nunique():,}",
        "ユニーク商品数": f"{metrics['product_key'].nunique():,}",
        "総サンプル数": f"{len(product_daily):,}",
        "全体ゼロ出荷比率": f"{(product_daily['actual_value'] == 0).mean() * 100:.1f}%",
    }
    avg_interval = metrics["avg_positive_interval"].dropna()
    if not avg_interval.empty:
        overview["平均正の出荷間隔"] = f"{avg_interval.mean():.2f}日"

    long_run = metrics[metrics["max_zero_streak"] >= 60]
    overview["60日以上ゼロ期間の商品数"] = (
        f"{len(long_run)}品 ({(len(long_run) / len(metrics)) * 100:.1f}% )"
        if len(metrics) > 0
        else "0品 (0.0%)"
    )
    half_share = metrics[metrics["zero_share"] >= 0.5]
    overview["ゼロ出荷比率50%以上の商品数"] = (
        f"{len(half_share)}品 ({(len(half_share) / len(metrics)) * 100:.1f}%)"
        if len(metrics) > 0
        else "0品 (0.0%)"
    )

    usage_zero_share = (
        metrics.groupby("usage_type_jp", observed=True)["zero_share"]
        .mean()
        .sort_values(ascending=False)
        .to_dict()
    )

    top_zero = metrics.sort_values("zero_share", ascending=False).head(5)[
        ["product_name", "zero_share", "avg_zero_streak", "mean_positive_demand"]
    ]
    top_longest = metrics.sort_values("max_zero_streak", ascending=False).head(5)[
        ["product_name", "max_zero_streak", "avg_zero_streak", "usage_type_jp"]
    ]

    return {
        "overview": overview,
        "usage_zero_share": usage_zero_share,
        "top_zero_products": top_zero,
        "top_longest_products": top_longest,
    }


def save_plot(fig: plt.Figure, filename: str) -> None:
    """図を保存してからクローズする。"""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"図を保存: {path}")


# --------------------------------------------------------------------------------------
# 可視化
# --------------------------------------------------------------------------------------
def plot_zero_share_top_products(metrics: pd.DataFrame) -> Insight | None:
    if metrics.empty:
        return None
    top = metrics.sort_values("zero_share", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["product_name"], top["zero_share"] * 100, color="#d62728")
    ax.set_xlabel("ゼロ出荷比率 (%)")
    ax.set_ylabel("商品名")
    ax.set_title("ゼロ出荷比率が高い上位20商品")
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center")
    save_plot(fig, "zero_share_top_products.png")
    focus = top.iloc[0]
    return Insight(
        "ゼロ出荷比率が高い商品",
        f"『{focus['product_name']}』は全期間の{focus['zero_share'] * 100:.1f}%がゼロ出荷であり、平均でも{focus['avg_zero_streak']:.1f}日連続で出荷が途切れている。"
    )


def plot_zero_run_length_distribution(sequences: pd.DataFrame) -> Insight | None:
    if sequences.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(sequences["length_days"], bins=30, color="#1f77b4", ax=ax)
    ax.set_title("ゼロ出荷連続日数の分布")
    ax.set_xlabel("連続ゼロ日数")
    ax.set_ylabel("出現回数")
    save_plot(fig, "zero_run_length_distribution.png")
    long_share = (sequences["length_days"] >= 14).mean()
    return Insight(
        "ゼロ期間の分布",
        f"ゼロ出荷が2週間以上続くケースは全体の{long_share * 100:.1f}%であり、需要の間欠性が顕著に表れている。"
    )


def plot_zero_run_box(metrics: pd.DataFrame) -> Insight | None:
    if metrics.empty:
        return None
    summary = (
        metrics.groupby("usage_type_jp", observed=True)["avg_zero_streak"]
        .agg(mean="mean", std="std", count="count")
        .reindex(["業務用", "家庭用"])
        .dropna(subset=["mean"])
    )
    if summary.empty:
        return None
    summary["std"] = summary["std"].fillna(0)
    colors = sns.color_palette("Set2", n_colors=len(summary))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        color=colors,
        capsize=8,
        edgecolor="black",
    )
    ax.set_xlabel("利用区分")
    ax.set_ylabel("平均連続ゼロ日数")
    ax.set_title("利用区分別の平均連続ゼロ日数（平均±標準偏差）")
    for bar, value in zip(bars, summary["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value:.1f}日",
            ha="center",
            va="bottom",
        )
    save_plot(fig, "zero_run_box_usage_type.png")
    pivot = (
        metrics.groupby("usage_type_jp", observed=True)["avg_zero_streak"]
        .mean()
        .sort_values(ascending=False)
    )
    top_usage = pivot.index[0]
    top_value = pivot.iloc[0]
    return Insight(
        "利用区分別の傾向",
        f"{top_usage}向け商品は平均連続ゼロ日数が{top_value:.1f}日と長く、需要創出のタイミングがより不規則である。"
    )


def plot_zero_heatmap(product_daily: pd.DataFrame) -> Insight | None:
    if product_daily.empty:
        return None
    monthly = (
        product_daily.assign(is_zero=product_daily["actual_value"] == 0)
        .groupby(["category_lvl1", "month"], observed=True)
        .agg(
            zero_days=("is_zero", "sum"),
            total_days=("is_zero", "count"),
        )
        .reset_index()
    )
    monthly["zero_ratio"] = monthly["zero_days"] / monthly["total_days"]
    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    pivot = monthly.pivot(index="category_lvl1", columns="month_dt", values="zero_ratio")
    pivot = pivot.sort_index(axis=1)
    pivot_pct = pivot * 100
    fig_width = max(10, pivot.shape[1] * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 2 + pivot.shape[0]))
    sns.heatmap(
        pivot_pct,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "ゼロ出荷比率 (%)"},
        )
    ax.set_title("カテゴリ別・月別のゼロ出荷比率")
    ax.set_xlabel("月")
    ax.set_ylabel("カテゴリ (大分類)")
    month_labels = [col.strftime("%Y-%m") for col in pivot.columns]
    ax.set_xticklabels(month_labels, rotation=45, ha="right")
    save_plot(fig, "zero_heatmap_category_month.png")
    hottest = monthly.sort_values("zero_ratio", ascending=False).iloc[0]
    return Insight(
        "カテゴリ×月の偏り",
        f"カテゴリ『{hottest['category_lvl1']}』では{hottest['month']}にゼロ出荷比率が{hottest['zero_ratio'] * 100:.1f}%まで上昇し、季節要因の影響が示唆される。"
    )


def plot_zero_timeline(product_daily: pd.DataFrame, metrics: pd.DataFrame) -> Insight | None:
    if product_daily.empty or metrics.empty:
        return None
    top_products = metrics.sort_values("max_zero_streak", ascending=False).head(3)
    if top_products.empty:
        return None
    all_dates = pd.date_range(
        product_daily["date"].min(), product_daily["date"].max(), freq="D"
    )
    date_nums = mdates.date2num(all_dates.to_pydatetime())
    cmap = ListedColormap(["#d62728", "#2ca02c"])
    fig, axes = plt.subplots(len(top_products), 1, figsize=(14, 1.8 * len(top_products)), sharex=True)
    if len(top_products) == 1:
        axes = [axes]
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_products.iterrows())):
        product_series = (
            product_daily[product_daily["product_key"] == row["product_key"]]
            .set_index("date")["actual_value"]
            .reindex(all_dates, fill_value=0)
        )
        binary = (product_series > 0).astype(int)
        ax.imshow(
            binary.values[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[date_nums[0] - 0.5, date_nums[-1] + 0.5, -0.5, 0.5],
        )
        ax.set_xlim(date_nums[0] - 0.5, date_nums[-1] + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        label = f"{row['product_name']} (最長ゼロ {int(row['max_zero_streak'])}日)"
        ax.set_title(label, loc="left", pad=10)
        if idx < len(axes) - 1:
            ax.set_xticks([])
    month_locator = mdates.MonthLocator(interval=2)
    month_formatter = mdates.DateFormatter("%Y-%m")
    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_locator(month_locator)
    axes[-1].xaxis.set_major_formatter(month_formatter)
    for label in axes[-1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    axes[-1].set_xlabel("日付")
    fig.suptitle("ゼロ出荷期間のタイムライン（赤=ゼロ出荷, 緑=出荷あり）", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, "zero_timeline.png")
    focus = top_products.iloc[0]
    return Insight(
        "最長ゼロ期間の可視化",
        f"『{focus['product_name']}』では最大{int(focus['max_zero_streak'])}日間の連続ゼロ出荷が観測され、販売計画の見直しが求められる。"
    )


def plot_interval_vs_zero(metrics: pd.DataFrame) -> Insight | None:
    filtered = metrics.dropna(subset=["avg_positive_interval"])
    if filtered.empty:
        return None
    plot_df = filtered.copy()
    plot_df["zero_share_pct"] = plot_df["zero_share"] * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="avg_positive_interval",
        y="zero_share_pct",
        hue="usage_type_jp",
        size="mean_positive_demand",
        palette="Set2",
        sizes=(50, 350),
        ax=ax,
    )
    ax.set_title("正の出荷間隔とゼロ出荷比率の関係")
    ax.set_xlabel("正の出荷間隔の平均日数")
    ax.set_ylabel("ゼロ出荷比率 (%)")
    ax.legend(title="利用区分 / 正出荷量", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_plot(fig, "interval_vs_zero.png")
    widest = filtered.sort_values("avg_positive_interval", ascending=False).iloc[0]
    return Insight(
        "間隔とゼロ比率の相関",
        f"平均間隔が{widest['avg_positive_interval']:.1f}日と長い『{widest['product_name']}』ではゼロ出荷比率が{widest['zero_share'] * 100:.1f}%に達し、補充頻度の適正化が必要と考えられる。"
    )


def _plot_zero_frequency_core(values: pd.Series, title: str, filename: str) -> float:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(values, bins=20, color="#9467bd", ax=ax, edgecolor="white")
    mean_val = values.mean()
    ax.axvline(mean_val, color="#ff7f0e", linestyle="--", label=f"平均 {mean_val:.1f}%")
    ax.set_title(title)
    ax.set_xlabel("ゼロ出荷の出現頻度 (%)")
    ax.set_ylabel("商品数")
    ax.legend()
    save_plot(fig, filename)
    return mean_val


def plot_zero_frequency_histogram(metrics: pd.DataFrame) -> Insight | None:
    if metrics.empty:
        return None
    values = metrics["zero_share_pct"].dropna()
    if values.empty:
        return None
    mean_val = _plot_zero_frequency_core(values, "商品別ゼロ出現頻度の分布", "zero_frequency_histogram.png")

    for usage in ["家庭用", "業務用"]:
        subset = metrics.loc[metrics["usage_type_jp"] == usage, "zero_share_pct"].dropna()
        if subset.empty:
            continue
        filename = f"zero_frequency_histogram_{'household' if usage == '家庭用' else 'business'}.png"
        title = f"商品別ゼロ出現頻度の分布（{usage}）"
        _plot_zero_frequency_core(subset, title, filename)

    high_share = (values >= 50).mean() * 100
    return Insight(
        "ゼロ出現頻度の分布",
        f"全商品の平均ゼロ出現頻度は{mean_val:.1f}%で、ゼロ頻度が50%以上の品は{high_share:.1f}%存在する。"
    )


# --------------------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------------------
def export_metrics(metrics: pd.DataFrame, sequences: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "zero_interval_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"間欠需要指標を出力: {metrics_path}")
    if not sequences.empty:
        seq_path = REPORT_DIR / "zero_sequences.csv"
        sequences.to_csv(seq_path, index=False)
        print(f"ゼロ出荷期間一覧を出力: {seq_path}")


def write_report(summary: Dict[str, object], insights: List[Insight]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "zero_interval_report.txt"
    lines: List[str] = []
    lines.append("=== データ概要 ===")
    for key, value in summary["overview"].items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("=== 利用区分別の平均ゼロ出荷比率 ===")
    usage_share = summary["usage_zero_share"]
    if usage_share:
        for usage, ratio in usage_share.items():
            lines.append(f"{usage}: 平均{ratio * 100:.1f}%")
    else:
        lines.append("該当データなし")
    lines.append("")
    lines.append("=== 主な示唆 ===")
    if insights:
        for insight in insights:
            lines.append(f"・{insight.title}: {insight.details}")
    else:
        lines.append("・特記すべき示唆は得られなかった。")
    lines.append("")
    lines.append("=== ゼロ出荷比率が高い商品 (上位5品) ===")
    top_zero = summary["top_zero_products"]
    if not top_zero.empty:
        for row in top_zero.itertuples(index=False):
            lines.append(
                f"・{row.product_name}: ゼロ出荷比率{row.zero_share * 100:.1f}%, "
                f"平均連続ゼロ日数{row.avg_zero_streak:.1f}日, 正出荷時平均{row.mean_positive_demand:.1f}"
            )
    else:
        lines.append("該当データなし")
    lines.append("")
    lines.append("=== 最長ゼロ期間を経験した商品 (上位5品) ===")
    top_longest = summary["top_longest_products"]
    if not top_longest.empty:
        for row in top_longest.itertuples(index=False):
            lines.append(
                f"・{row.product_name}: 最長{int(row.max_zero_streak)}日間のゼロ出荷 "
                f"(平均連続ゼロ{row.avg_zero_streak:.1f}日, 利用区分: {row.usage_type_jp})"
            )
    else:
        lines.append("該当データなし")
    lines.append("")
    lines.append("=== 推奨アクション例 ===")
    lines.append("・ゼロ出荷期間が長い商品に対しては販促・キャンペーンを集中させ、発注間隔の短縮を図る。")
    lines.append("・ゼロ出荷比率が高い商品群へはCroston系など間欠需要モデルの適用を検討し、安全在庫水準を最適化する。")
    lines.append("・カテゴリ×月でゼロ出荷比率が跳ね上がるタイミングを販売イベントや季節施策と照合し、計画修正に活用する。")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"レポートを出力: {report_path}")


# --------------------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------------------
def main() -> None:
    ensure_font()
    product_map = load_product_name_map()
    raw_df = pd.read_parquet(DATA_PATH)
    prepared_df = prepare_dataframe(raw_df, product_map)
    product_daily = aggregate_product_daily(prepared_df)
    zero_sequences = compute_zero_sequences(product_daily)
    zero_metrics = compute_zero_metrics(product_daily, zero_sequences)
    summary = summarize_zero_metrics(product_daily, zero_metrics)

    insights: List[Insight] = []

    insight = plot_zero_share_top_products(zero_metrics)
    if insight is not None:
        insights.append(insight)

    insight = plot_zero_run_length_distribution(zero_sequences)
    if insight is not None:
        insights.append(insight)

    insight = plot_zero_run_box(zero_metrics)
    if insight is not None:
        insights.append(insight)

    insight = plot_zero_heatmap(product_daily)
    if insight is not None:
        insights.append(insight)

    insight = plot_zero_timeline(product_daily, zero_metrics)
    if insight is not None:
        insights.append(insight)

    insight = plot_interval_vs_zero(zero_metrics)
    if insight is not None:
        insights.append(insight)

    insight = plot_zero_frequency_histogram(zero_metrics)
    if insight is not None:
        insights.append(insight)

    export_metrics(zero_metrics, zero_sequences)
    write_report(summary, insights)


if __name__ == "__main__":
    main()
