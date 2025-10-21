#!/usr/bin/env python3
"""
商品レベル予測モデル学習・逐次推論スクリプト

ポイント
--------
- `generate_features_product_level.py` が作成した逐次更新対応特徴量を読み込み、
  XGBoost モデルを訓練する。
- 推論時は `FeatureState` を用いて日次で特徴量を更新しながら前方に予測を進める。
- 生成した予測値は work/data 以下に保存し、可視化は
  /home/ubuntu/yamasa2/work/vis_2024-12-31_3 に出力する。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from pandas.tseries.offsets import MonthBegin, MonthEnd

from generate_features_product_level import (
    FEATURE_COLUMNS,
    FeatureState,
    MIN_HISTORY,
    STATIC_INFO_FILENAME,
    FEATURE_FILENAME_LATEST,
    load_aggregated_data,
    ensure_directory,
)


DEFAULT_BASE_DIR = "/home/ubuntu/yamasa2/work/data"
DEFAULT_VIS_DIR = "/home/ubuntu/yamasa2/work/vis_2024-12-31_3"

PREDICTIONS_DIRNAME = "predictions"
MODELS_DIRNAME = "models"
OUTPUT_DIRNAME = "output"
EPSILON = 1e-6
SOFT_CLIP_MULTIPLIER = 1.5
WEEKEND_WHITELIST: set[str] = set()


def _parse_month(month_str: str) -> pd.Timestamp:
    if month_str is None:
        raise ValueError("Month string must not be None.")
    return pd.to_datetime(f"{month_str}-01")


def _compute_horizon_months(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


# --------------------------------------------------------------------------------------
# ロード系
# --------------------------------------------------------------------------------------

def load_training_features(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "features", FEATURE_FILENAME_LATEST)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training feature file not found: {path}")
    df = pd.read_parquet(path)
    df["file_date"] = pd.to_datetime(df["file_date"])
    return df


def load_static_info(base_dir: str) -> Dict[str, Dict[str, float]]:
    path = os.path.join(base_dir, "features", STATIC_INFO_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Static info file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    static_info = {}
    for material_key, info in data.items():
        # JSON 上は文字列キーになっている月別統計を int に戻す
        month_mean_map = {int(k): float(v) for k, v in info.get("month_mean_map", {}).items()}
        month_std_map = {int(k): float(v) for k, v in info.get("month_std_map", {}).items()}
        week_mean_map = {int(k): float(v) for k, v in info.get("week_mean_map", {}).items()}
        week_std_map = {int(k): float(v) for k, v in info.get("week_std_map", {}).items()}
        weekday_mean_map = {int(k): float(v) for k, v in info.get("weekday_mean_map", {}).items()}

        static_info[material_key] = {
            "material_idx": float(info.get("material_idx", 0.0)),
            "product_mean_f": float(info.get("product_mean_f", 0.0)),
            "product_std_f": float(info.get("product_std_f", 0.0)),
            "product_median_f": float(info.get("product_median_f", 0.0)),
            "product_min_f": float(info.get("product_min_f", 0.0)),
            "product_max_f": float(info.get("product_max_f", 0.0)),
            "volume_segment_f": float(info.get("volume_segment_f", 0.0)),
            "month_mean_map": month_mean_map,
            "month_std_map": month_std_map,
            "week_mean_map": week_mean_map,
            "week_std_map": week_std_map,
            "weekday_mean_map": weekday_mean_map,
        }

    return static_info


def ensure_forecast_coverage(
    aggregated_df: pd.DataFrame,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    """予測対象期間まで日次カレンダーが連続するよう補完する。"""

    frames = []
    for material_key, group in aggregated_df.groupby("material_key"):
        group = group.sort_values("file_date")
        start_date = group["file_date"].min()
        max_date = max(group["file_date"].max(), forecast_end)
        full_dates = pd.date_range(start=start_date, end=max_date, freq="D")

        base = pd.DataFrame({"file_date": full_dates})
        base = base.merge(group[["file_date", "actual_value", "product_key"]], on="file_date", how="left")
        base["material_key"] = material_key
        base["product_key"] = base["product_key"].fillna(material_key)
        base["actual_value"] = base["actual_value"].fillna(0.0)

        frames.append(base[["material_key", "product_key", "file_date", "actual_value"]])

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["material_key", "file_date"], inplace=True)
    return result


# --------------------------------------------------------------------------------------
# 訓練関連
# --------------------------------------------------------------------------------------

def split_train_validation(
    features_df: pd.DataFrame,
    train_end_date: str,
    validation_days: int = 28,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """時間に従う単純な学習/検証分割。"""
    cutoff = pd.to_datetime(train_end_date)
    val_start = cutoff - timedelta(days=validation_days)

    val_mask = features_df["file_date"] >= val_start
    if val_mask.sum() == 0:
        # 極端に短い場合は14日で再試行
        val_start = cutoff - timedelta(days=max(14, validation_days // 2))
        val_mask = features_df["file_date"] >= val_start

    train_mask = ~val_mask
    if train_mask.sum() == 0:
        # 学習データが消えてしまう場合は全データを学習に使用
        train_mask[:] = True
        val_mask[:] = False

    train_df = features_df.loc[train_mask].copy()
    val_df = features_df.loc[val_mask].copy()

    return train_df, val_df


def to_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].astype(np.float32).values
    y = df["actual_value"].astype(np.float32).values
    return X, y


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    X_train, y_train = to_matrix(train_df, feature_cols)
    X_val, y_val = to_matrix(val_df, feature_cols) if not val_df.empty else (None, None)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    if X_val is not None and X_val.size > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)

    metrics = {}
    if X_val is not None and X_val.size > 0:
        y_pred_val = model.predict(X_val)
        metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
        metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        smape = (
            2 * np.abs(y_pred_val - y_val)
            / (np.abs(y_pred_val) + np.abs(y_val) + EPSILON)
        ).mean() * 100
        wape = np.sum(np.abs(y_pred_val - y_val)) / (np.sum(np.abs(y_val)) + EPSILON)
        metrics["val_mape"] = float(smape)
        metrics["val_accuracy"] = float(100 - smape)
        metrics["val_wape"] = float(wape)

    return model, metrics


# --------------------------------------------------------------------------------------
# 逐次推論
# --------------------------------------------------------------------------------------

def sequential_forecast(
    model: XGBRegressor,
    aggregated_df: pd.DataFrame,
    static_info_map: Dict[str, Dict[str, float]],
    train_cutoff_date: pd.Timestamp,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    feature_cols: List[str],
) -> pd.DataFrame:
    predictions: List[Dict[str, float]] = []

    date_range = pd.date_range(start=forecast_start, end=forecast_end, freq="D")

    for material_key, group in aggregated_df.groupby("material_key"):
        if material_key not in static_info_map:
            continue

        group = group.sort_values("file_date")
        static_info = static_info_map[material_key]
        state = FeatureState(min_history=MIN_HISTORY)

        history_rows = group[group["file_date"] <= train_cutoff_date]
        if history_rows.empty:
            continue

        for row in history_rows.itertuples(index=False):
            state.add_observation(row.file_date, float(row.actual_value))

        actual_series = group.set_index("file_date")["actual_value"]

        for current_date in date_range:
            feature_row = state.build_feature_row(current_date, static_info)
            if feature_row is None:
                y_pred = float(static_info.get("product_mean_f", 0.0))
            else:
                vector = np.array([feature_row[col] for col in feature_cols], dtype=np.float32).reshape(1, -1)
                y_pred = float(model.predict(vector)[0])

            if feature_row:
                peak_actual = feature_row.get("recent_peak_actual_f", 0.0)
                if peak_actual and peak_actual > 0:
                    cap = float(peak_actual) * SOFT_CLIP_MULTIPLIER
                    if y_pred > cap:
                        delta = y_pred - cap
                        scale = cap if cap > EPSILON else 1.0
                        y_pred = cap + scale * math.tanh(delta / (scale + EPSILON))

            if current_date.weekday() >= 5 and material_key not in WEEKEND_WHITELIST:
                y_pred = 0.0

            y_pred = max(y_pred, 0.0)
            actual_value = float(actual_series.get(current_date, 0.0))

            predictions.append(
                {
                    "material_key": material_key,
                    "date": current_date,
                    "predicted": y_pred,
                    "actual": actual_value,
                }
            )

            state.add_observation(current_date, y_pred, is_actual=False)

    pred_df = pd.DataFrame(predictions)
    if pred_df.empty:
        raise RuntimeError(
            "No predictions were generated. Check forecast range or training cut-off date."
        )

    pred_df.sort_values(["material_key", "date"], inplace=True)
    return pred_df


# --------------------------------------------------------------------------------------
# 評価・保存
# --------------------------------------------------------------------------------------

def compute_overall_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["total_actual"] = float(pred_df["actual"].sum())
    metrics["total_predicted"] = float(pred_df["predicted"].sum())
    metrics["mae"] = float(mean_absolute_error(pred_df["actual"], pred_df["predicted"]))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(pred_df["actual"], pred_df["predicted"])))

    smape = (
        2 * np.abs(pred_df["predicted"] - pred_df["actual"])
        / (np.abs(pred_df["predicted"]) + np.abs(pred_df["actual"]) + EPSILON)
    ).mean() * 100
    metrics["mape"] = float(smape)
    metrics["accuracy"] = float(100 - smape)
    wape = np.sum(np.abs(pred_df["predicted"] - pred_df["actual"])) / (np.sum(np.abs(pred_df["actual"])) + EPSILON)
    metrics["wape"] = float(wape)

    ratio = metrics["total_predicted"] / metrics["total_actual"] if metrics["total_actual"] > 0 else float("nan")
    metrics["pred_to_actual_ratio"] = float(ratio)

    return metrics


def compute_per_product_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for material_key, group in pred_df.groupby("material_key"):
        mae = mean_absolute_error(group["actual"], group["predicted"])
        rmse = np.sqrt(mean_squared_error(group["actual"], group["predicted"]))
        actual_sum = group["actual"].sum()
        predicted_sum = group["predicted"].sum()

        smape = (
            2 * np.abs(group["predicted"] - group["actual"])
            / (np.abs(group["predicted"]) + np.abs(group["actual"]) + EPSILON)
        ).mean() * 100
        accuracy = 100 - smape
        wape = np.sum(np.abs(group["predicted"] - group["actual"])) / (np.sum(np.abs(group["actual"])) + EPSILON)

        records.append(
            {
                "material_key": material_key,
                "n_samples": int(len(group)),
                "actual_sum": float(actual_sum),
                "predicted_sum": float(predicted_sum),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(smape),
                "wape": float(wape),
                "accuracy": float(accuracy),
            }
        )

    summary_df = pd.DataFrame(records).sort_values("accuracy", ascending=False)
    return summary_df


def extract_feature_importance(model: XGBRegressor, feature_cols: List[str]) -> pd.DataFrame:
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    records = [
        {"feature": feature, "gain": importance.get(feature, 0.0)}
        for feature in feature_cols
    ]
    df = pd.DataFrame(records).sort_values("gain", ascending=False)
    return df


def save_outputs(
    base_dir: str,
    train_end_date: str,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    pred_df: pd.DataFrame,
    per_product_df: pd.DataFrame,
    overall_metrics: Dict[str, float],
    feature_importance_df: pd.DataFrame,
) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_dir = os.path.join(base_dir, PREDICTIONS_DIRNAME)
    models_dir = os.path.join(base_dir, MODELS_DIRNAME)
    output_dir = os.path.join(base_dir, OUTPUT_DIRNAME)
    ensure_directory(predictions_dir)
    ensure_directory(models_dir)
    ensure_directory(output_dir)

    # 1. 予測詳細
    pred_latest = os.path.join(predictions_dir, "product_level_predictions_latest.parquet")
    pred_dated = os.path.join(predictions_dir, f"product_level_predictions_{timestamp}.parquet")
    for path in (pred_latest, pred_dated):
        pred_df.to_parquet(path, index=False)

    pred_csv = os.path.join(output_dir, "product_level_predictions_latest.csv")
    pred_df.to_csv(pred_csv, index=False)

    # 2. サマリー
    summary_latest = os.path.join(predictions_dir, "product_level_summary_latest.parquet")
    summary_dated = os.path.join(predictions_dir, f"product_level_summary_{timestamp}.parquet")
    for path in (summary_latest, summary_dated):
        per_product_df.to_parquet(path, index=False)

    summary_csv = os.path.join(output_dir, "product_level_summary_latest.csv")
    per_product_df.to_csv(summary_csv, index=False)

    # 3. メトリクス
    metrics_payload = {
        "timestamp": timestamp,
        "train_end_date": train_end_date,
        "forecast_start": forecast_start.strftime("%Y-%m-%d"),
        "forecast_end": forecast_end.strftime("%Y-%m-%d"),
        "metrics": overall_metrics,
    }
    metrics_latest = os.path.join(models_dir, "product_level_metrics_latest.json")
    metrics_dated = os.path.join(models_dir, f"product_level_metrics_{timestamp}.json")
    for path in (metrics_latest, metrics_dated):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    # 4. 特徴量重要度
    fi_latest = os.path.join(models_dir, "feature_importance_latest.csv")
    fi_dated = os.path.join(models_dir, f"feature_importance_{timestamp}.csv")
    for path in (fi_latest, fi_dated):
        feature_importance_df.to_csv(path, index=False)

    return {
        "predictions_parquet": pred_latest,
        "predictions_csv": pred_csv,
        "summary_parquet": summary_latest,
        "summary_csv": summary_csv,
        "metrics_json": metrics_latest,
        "feature_importance_csv": fi_latest,
    }


# --------------------------------------------------------------------------------------
# 可視化
# --------------------------------------------------------------------------------------

def run_visualization(
    predictions_csv: str,
    output_dir: str,
    historical_data_path: str,
    horizon_months: int = 4,
    history_months: int = 3,
) -> None:
    """
    PredictionVisualizer を利用してグラフ・サマリを出力する。
    """
    from visualize_predictions import PredictionVisualizer

    visualizer = PredictionVisualizer(
        input_file=predictions_csv,
        output_dir=output_dir,
        historical_data_file=historical_data_path,
        horizon_months=horizon_months,
        history_months=history_months,
    )
    pred_df = visualizer.load_prediction_data()
    hist_df = visualizer.load_historical_data()
    visualizer.generate_reports(pred_df, hist_df)


# --------------------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------------------

def main(
    base_dir: str = DEFAULT_BASE_DIR,
    train_end_date: str = "2024-12-31",
    forecast_month_start: Optional[str] = None,
    forecast_month_end: Optional[str] = None,
    visualization_dir: str = DEFAULT_VIS_DIR,
    disable_visualization: bool = False,
) -> None:
    print("=" * 60)
    print("Product-level Model Training & Sequential Prediction")
    print(f"Timestamp         : {datetime.now()}")
    print(f"Base directory    : {base_dir}")
    print(f"Train cutoff date : {train_end_date}")
    print(f"Visualization dir : {visualization_dir}")
    print("=" * 60)

    train_cutoff_dt = pd.to_datetime(train_end_date)
    if forecast_month_start:
        forecast_start_dt = _parse_month(forecast_month_start)
    else:
        forecast_start_dt = (train_cutoff_dt + MonthBegin())

    if forecast_month_end:
        forecast_end_dt = _parse_month(forecast_month_end) + MonthEnd(0)
    else:
        forecast_end_dt = forecast_start_dt + MonthEnd(0)

    if forecast_end_dt < forecast_start_dt:
        raise ValueError("forecast_month_end must not be earlier than forecast_month_start.")

    print(f"Forecast month    : {forecast_start_dt.date()} ~ {forecast_end_dt.date()}")

    features_df = load_training_features(base_dir)
    features_df = features_df[features_df["file_date"] <= train_cutoff_dt].copy()
    if features_df.empty:
        raise ValueError("Training features are empty after applying train cutoff date.")

    static_info_map = load_static_info(base_dir)
    feature_cols = [col for col in FEATURE_COLUMNS if col in features_df.columns]

    print(f"Training rows     : {len(features_df):,}")
    print(f"Feature columns   : {len(feature_cols)} -> {feature_cols}")

    train_df, val_df = split_train_validation(features_df, train_cutoff_dt.strftime("%Y-%m-%d"))
    print(f"Train split       : {len(train_df):,} rows")
    print(f"Validation split  : {len(val_df):,} rows")

    model, val_metrics = train_model(train_df, val_df, feature_cols)
    if val_metrics:
        print("\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    aggregated_df = load_aggregated_data(base_dir)
    aggregated_df["file_date"] = pd.to_datetime(aggregated_df["file_date"])
    aggregated_df = ensure_forecast_coverage(aggregated_df, forecast_end_dt)

    pred_df = sequential_forecast(
        model,
        aggregated_df,
        static_info_map,
        train_cutoff_dt,
        forecast_start_dt,
        forecast_end_dt,
        feature_cols,
    )

    overall_metrics = compute_overall_metrics(pred_df)
    per_product_df = compute_per_product_metrics(pred_df)
    feature_importance_df = extract_feature_importance(model, feature_cols)

    print("\nOverall metrics (sequential forecast):")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    saved_paths = save_outputs(
        base_dir,
        train_end_date,
        forecast_start_dt,
        forecast_end_dt,
        pred_df,
        per_product_df,
        overall_metrics,
        feature_importance_df,
    )

    print("\nSaved artifacts:")
    for label, path in saved_paths.items():
        print(f"  {label}: {path}")

    if not disable_visualization:
        horizon_months = _compute_horizon_months(forecast_start_dt, forecast_end_dt)
        print("\nGenerating visualizations...")
        ensure_directory(visualization_dir)
        historical_path = os.path.join(base_dir, "prepared", "df_confirmed_order_input_yamasa_fill_zero.parquet")
        run_visualization(
            predictions_csv=saved_paths["predictions_csv"],
            output_dir=visualization_dir,
            historical_data_path=historical_path,
            horizon_months=horizon_months,
            history_months=3,
        )
        print(f"Visualizations created in: {visualization_dir}")

    print("\nTraining and forecasting completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train product-level model and generate sequential forecasts.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"データベースディレクトリ (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="2024-12-31",
        help="学習データの終了日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forecast-month-start",
        type=str,
        default=None,
        help="予測対象月の開始 (YYYY-MM)。指定しない場合は train_end_date の翌月",
    )
    parser.add_argument(
        "--forecast-month-end",
        type=str,
        default=None,
        help="予測対象月の終了 (YYYY-MM)。指定しない場合は開始月のみを予測",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=DEFAULT_VIS_DIR,
        help=f"可視化出力先ディレクトリ (default: {DEFAULT_VIS_DIR})",
    )
    parser.add_argument(
        "--disable-visualization",
        action="store_true",
        help="可視化生成をスキップする場合に指定",
    )

    args = parser.parse_args()
    main(
        base_dir=args.base_dir,
        train_end_date=args.train_end_date,
        forecast_month_start=args.forecast_month_start,
        forecast_month_end=args.forecast_month_end,
        visualization_dir=args.visualization_dir,
        disable_visualization=args.disable_visualization,
    )
