#!/usr/bin/env python3
"""
商品レベル予測結果の可視化スクリプト（逐次更新対応版）

主な変更点
-----------
- 日付リインデックス時にゼロ埋めを行わず、欠測は欠測のまま保持
- 予測値・実績値の誤差指標を読み込み時に付与
- サマリ（CSV/JSON/TXT）とグラフ出力のユーティリティ `generate_reports` を提供
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"


class PredictionVisualizer:
    """予測結果の可視化・サマリ生成を担当するクラス"""

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        historical_data_file: Optional[str] = None,
        horizon_months: Optional[int] = 6,
        history_months: int = 3,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.historical_data_file = Path(historical_data_file) if historical_data_file else None
        self.horizon_months = horizon_months
        self.history_months = max(history_months, 0)

        self.graph_dir = self.output_dir / "graph"
        self.summary_dir = self.output_dir / "summary"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # データ読み込み
    # ------------------------------------------------------------------
    def load_prediction_data(self) -> pd.DataFrame:
        """予測結果を読み込み、誤差列を付与する。"""
        print(f"予測結果を読み込み: {self.input_file}")
        if not self.input_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.input_file}")

        if self.input_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.input_file)
        else:
            df = pd.read_csv(self.input_file)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["material_key", "date"])
        df["error"] = df["predicted"] - df["actual"]
        df["abs_error"] = df["error"].abs()
        denom = np.abs(df["predicted"]) + np.abs(df["actual"]) + 1e-6
        df["smape_pct"] = 200 * df["abs_error"] / denom
        # 顧客説明用の予測誤差（実績が0の場合は0%）
        df["simple_error_pct"] = np.where(
            df["actual"] == 0,
            0.0,
            (df["abs_error"] / np.abs(df["actual"])) * 100.0,
        )
        df["predict_year_month"] = df["date"].dt.strftime("%Y-%m")
        df["relative_error"] = df["smape_pct"] / 100.0

        print(f"  行数: {len(df):,}, 期間: {df['date'].min()} ~ {df['date'].max()}")
        return df

    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """過去実績データを必要に応じて読み込む。"""
        if self.historical_data_file and self.historical_data_file.exists():
            print(f"過去データを読み込み: {self.historical_data_file}")
            hist_df = pd.read_parquet(self.historical_data_file)
            hist_df["file_date"] = pd.to_datetime(hist_df["file_date"])
            return hist_df
        print("過去データファイルが見つからないため、履歴表示をスキップします。")
        return None

    # ------------------------------------------------------------------
    # 精度指標
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics_for_product(product_df: pd.DataFrame) -> Dict[str, float]:
        mae = float(product_df["abs_error"].mean())
        rmse = float(np.sqrt(np.mean(product_df["error"] ** 2)))
        actual_sum = float(product_df["actual"].sum())
        predicted_sum = float(product_df["predicted"].sum())
        n_samples = int(len(product_df))

        smape = float(product_df["smape_pct"].mean())
        accuracy = float(100 - smape)
        simple_error = float(product_df["simple_error_pct"].mean())
        wape = float(np.sum(product_df["abs_error"]) / (np.sum(np.abs(product_df["actual"])) + 1e-6) * 100)

        return {
            "mape": smape,
            "mae": mae,
            "rmse": rmse,
            "accuracy": accuracy,
            "simple_error": simple_error,
            "wape": wape,
            "n_samples": n_samples,
            "actual_sum": actual_sum,
            "predicted_sum": predicted_sum,
        }

    def calculate_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for material_key, group in df.groupby("material_key"):
            metrics[material_key] = self._compute_metrics_for_product(group)
        return metrics

    # ------------------------------------------------------------------
    # グラフ生成
    # ------------------------------------------------------------------
    def create_product_graph(
        self,
        pred_df: pd.DataFrame,
        product_code: str,
        hist_df: Optional[pd.DataFrame] = None,
    ) -> None:
        product_pred_df = pred_df[pred_df["material_key"] == product_code].copy()
        if product_pred_df.empty:
            return

        product_pred_df = product_pred_df.sort_values("date")
        pred_start = product_pred_df["date"].min()

        if self.horizon_months is not None:
            earliest = product_pred_df["date"].min()
            cutoff = earliest + pd.DateOffset(months=self.horizon_months)
            product_pred_df = product_pred_df[product_pred_df["date"] <= cutoff]

        plt.figure(figsize=(14, 7))

        if hist_df is not None and product_code in hist_df["material_key"].values:
            product_hist_df = hist_df[hist_df["material_key"] == product_code].copy()
            product_hist_df = product_hist_df.sort_values("file_date")
            hist_start = pred_start - pd.DateOffset(months=self.history_months)
            product_hist_df = product_hist_df[
                (product_hist_df["file_date"] >= hist_start)
                & (product_hist_df["file_date"] < pred_start)
            ]
            if not product_hist_df.empty:
                plt.plot(
                    product_hist_df["file_date"],
                    product_hist_df["actual_value"],
                    color="lightgray",
                    linewidth=1.5,
                    label="Historical Actual (6 months)",
                    alpha=0.5,
                )

        plt.plot(
            product_pred_df["date"],
            product_pred_df["actual"],
            color="gray",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Test Period Actual",
            alpha=0.7,
        )
        plt.plot(
            product_pred_df["date"],
            product_pred_df["predicted"],
            color="red",
            linewidth=2,
            marker="s",
            markersize=5,
            label="Predicted",
            alpha=0.7,
        )

        title = f"Product: {product_code} - Actual vs Predicted"
        valid_errors = product_pred_df["smape_pct"].dropna()
        if not valid_errors.empty:
            title += f" (Accuracy: {100 - valid_errors.mean():.1f}%, Err: {product_pred_df['simple_error_pct'].mean():.1f}%)"

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Demand", fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.xlim([pred_start - pd.DateOffset(months=self.history_months), product_pred_df["date"].max()])

        output_path = self.graph_dir / f"{product_code}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"  グラフを出力: {output_path}")

    # ------------------------------------------------------------------
    # サマリ保存
    # ------------------------------------------------------------------
    def save_summary(
        self,
        pred_df: pd.DataFrame,
        accuracy_by_product: Dict[str, Dict[str, float]],
    ) -> None:
        summary_csv = self.summary_dir / "accuracy_summary.csv"
        summary_json = self.summary_dir / "accuracy_summary.json"
        summary_txt = self.summary_dir / "accuracy_summary.txt"

        df_summary = pd.DataFrame.from_dict(accuracy_by_product, orient="index")
        df_summary.index.name = "material_key"
        df_summary.reset_index(inplace=True)
        df_summary.sort_values("accuracy", ascending=False, inplace=True)
        df_summary.to_csv(summary_csv, index=False)
        df_summary.to_json(summary_json, orient="records", indent=2, force_ascii=False)

        overall_mae = float(pred_df["abs_error"].mean())
        overall_rmse = float(np.sqrt(np.mean(pred_df["error"] ** 2)))
        overall_smape = float(pred_df["smape_pct"].mean())
        overall_simple_error = float(pred_df["simple_error_pct"].mean())
        overall_wape = float(np.sum(pred_df["abs_error"]) / (np.sum(np.abs(pred_df["actual"])) + 1e-6) * 100)

        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("Accuracy Summary Report\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Overall MAE : {overall_mae:.2f}\n")
            f.write(f"Overall RMSE: {overall_rmse:.2f}\n")
            f.write(f"Overall sMAPE: {overall_smape:.2f}% "
                    f"(Accuracy {100 - overall_smape:.2f}%)\n")
            f.write(f"Overall Simple Error: {overall_simple_error:.2f}%\n")
            f.write(f"Overall WAPE : {overall_wape:.2f}%\n")
            f.write("\nTop 10 products by accuracy:\n")

            for _, row in df_summary.head(10).iterrows():
                f.write(
                    f"- {row['material_key']}: Accuracy {row['accuracy']:.2f}%, "
                    f"sMAPE {row['mape']:.2f}%, SimpleErr {row.get('simple_error', float('nan')):.2f}%, WAPE {row.get('wape', np.nan):.2f}%, "
                    f"MAE {row['mae']:.2f}, RMSE {row['rmse']:.2f}, Samples {row['n_samples']}\n"
                )

        print(f"サマリを保存: {summary_csv}, {summary_json}, {summary_txt}")

    # ------------------------------------------------------------------
    # 総合レポート
    # ------------------------------------------------------------------
    def generate_reports(
        self,
        pred_df: pd.DataFrame,
        hist_df: Optional[pd.DataFrame] = None,
    ) -> None:
        accuracy_by_product = self.calculate_accuracy_metrics(pred_df)
        self.save_summary(pred_df, accuracy_by_product)

        print("グラフ生成中...")
        for product_code in pred_df["material_key"].unique():
            self.create_product_graph(pred_df, product_code, hist_df)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize product-level predictions.")
    parser.add_argument("--input-file", required=True, help="予測結果ファイル（CSVまたはparquet）")
    parser.add_argument("--output-dir", required=True, help="出力先ディレクトリ")
    parser.add_argument("--historical-file", help="過去データparquetファイル（任意）")
    parser.add_argument(
        "--horizon-months",
        type=int,
        default=6,
        help="グラフで表示する予測期間（月数）。全期間を表示する場合は負値を指定",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        default=3,
        help="グラフの表示開始を予測開始日の何か月前にするか",
    )
    args = parser.parse_args()

    horizon = args.horizon_months if args.horizon_months >= 0 else None
    visualizer = PredictionVisualizer(
        input_file=args.input_file,
        output_dir=args.output_dir,
        historical_data_file=args.historical_file,
        horizon_months=horizon,
        history_months=args.history_months,
    )

    pred_df = visualizer.load_prediction_data()
    hist_df = visualizer.load_historical_data()
    visualizer.generate_reports(pred_df, hist_df)


if __name__ == "__main__":
    main()
