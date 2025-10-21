#!/usr/bin/env python3
"""
商品レベルの特徴量生成（逐次更新対応版）

目的:
    - 学習データ（train_end_date以前）のみを用いたリークレスな特徴量を生成する
    - 予測時に逐次的に更新できる特徴量セットを定義・共有する
    - 特徴量仕様・マッピング情報を JSON として保存し、学習・推論で再利用する

出力:
    - work/data/features/product_level_features_latest.parquet
    - work/data/features/product_level_features_YYYYmmdd_HHMMSS.parquet
    - work/data/features/product_static_info.json

このスクリプトは `FeatureState` クラスを公開し、学習・推論の両方で同一ロジックの
特徴量を利用できるようにしている。推論側では、このクラスに予測済み需要を連続的に
投入することで、マルチステップ予測中もラグ/ローリング指標が更新される。
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import jpholiday
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset, MonthEnd


# --------------------------------------------------------------------------------------
# 定数定義
# --------------------------------------------------------------------------------------

DEFAULT_BASE_DIR = "/home/ubuntu/yamasa2/work/data"
FEATURE_FILENAME_LATEST = "product_level_features_latest.parquet"
FEATURE_FILENAME_TEMPLATE = "product_level_features_{timestamp}.parquet"
STATIC_INFO_FILENAME = "product_static_info.json"
AGGREGATED_INPUT_FILENAME = "df_confirmed_order_input_yamasa_fill_zero.parquet"

MIN_HISTORY = 28
MAX_HISTORY = 180
EWM_SPANS = (7, 14, 30)
ROLLING_WINDOWS = (3, 7, 14, 28)
STD_WINDOWS = (7, 14, 28)
NONZERO_WINDOWS = (7, 14)
TREND_LAGS = (7, 14)
VOLUME_THRESHOLD = 5000
WEEKEND_LOOKBACK_DAYS = 28
WEEKEND_ZERO_THRESHOLD = 0.1


CALENDAR_FEATURES = [
    "dow",
    "is_weekend",
    "is_business_day_f",
    "month",
    "quarter",
    "day_of_month",
    "iso_week",
    "is_month_start",
    "is_month_end",
    "days_to_next_holiday_f",
    "days_from_prev_holiday_f",
    "is_payday_window_f",
    "fiscal_period_f",
]

STATEFUL_FEATURES = [
    "lag_1_f",
    "lag_2_f",
    "lag_3_f",
    "lag_7_f",
    "lag_14_f",
    "lag_28_f",
    "rolling_mean_3_f",
    "rolling_mean_7_f",
    "rolling_mean_14_f",
    "rolling_mean_28_f",
    "rolling_std_7_f",
    "rolling_std_14_f",
    "rolling_std_28_f",
    "rolling_min_7_f",
    "rolling_max_7_f",
    "rolling_median_7_f",
    "rolling_median_14_f",
    "rolling_iqr_14_f",
    "rolling_cv_14_f",
    "peak_to_mean_ratio_14_f",
    "ewm_mean_7_f",
    "ewm_mean_14_f",
    "ewm_mean_30_f",
    "recent_nonzero_ratio_7_f",
    "recent_nonzero_ratio_14_f",
    "recent_nonzero_run_length_7_f",
    "recent_nonzero_run_length_14_f",
    "zero_run_length_7_f",
    "zero_run_length_14_f",
    "days_since_last_obs_f",
    "days_since_last_nonzero_f",
    "gap_business_days_f",
    "cumulative_mean_f",
    "cumulative_std_f",
    "trend_7_ratio_f",
    "trend_14_ratio_f",
    "trend_28_slope_f",
    "month_over_month_ratio_f",
    "seasonal_month_mean_f",
    "seasonal_month_std_f",
    "seasonal_week_mean_f",
    "seasonal_week_std_f",
    "cumulative_pred_to_actual_ratio_f",
    "recent_bias_ratio_14_f",
    "weekend_zero_flag_f",
    "weekend_nonzero_rate_f",
    "dow_nonzero_rate_f_0",
    "dow_nonzero_rate_f_1",
    "dow_nonzero_rate_f_2",
    "dow_nonzero_rate_f_3",
    "dow_nonzero_rate_f_4",
    "dow_nonzero_rate_f_5",
    "dow_nonzero_rate_f_6",
]

STATIC_FEATURES = [
    "material_idx",
    "product_mean_f",
    "product_std_f",
    "product_median_f",
    "product_min_f",
    "product_max_f",
    "volume_segment_f",
]

FEATURE_COLUMNS: List[str] = STATEFUL_FEATURES + STATIC_FEATURES + CALENDAR_FEATURES


# --------------------------------------------------------------------------------------
# ヘルパー関数
# --------------------------------------------------------------------------------------

def is_business_day(date: pd.Timestamp) -> int:
    """営業日判定（土日祝を除外）。"""
    if pd.isna(date):
        return 0
    if date.weekday() >= 5:
        return 0
    if jpholiday.is_holiday(date):
        return 0
    return 1


def count_business_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    2つの日付の間（start, end を除く）に含まれる営業日数をカウントする。
    """
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current < end:
        count += is_business_day(current)
        current += timedelta(days=1)
    return count


def safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    """ゼロ割を避けるための安全な比率計算。"""
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return default
    return float(numerator) / float(denominator)


def business_days_distance(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """start から end までの営業日差分（end 側方向を正とする）。"""
    if pd.isna(start) or pd.isna(end) or start == end:
        return 0
    step = 1 if start < end else -1
    current = start
    distance = 0
    while current != end:
        current += timedelta(days=step)
        if is_business_day(current):
            distance += step
    return distance


def days_to_next_holiday(date: pd.Timestamp, max_search: int = 30) -> float:
    """次の休日日（祝日または週末）までの日数。見つからなければ max_search+1 を返す。"""
    for offset in range(1, max_search + 1):
        candidate = date + timedelta(days=offset)
        if not is_business_day(candidate):
            return float(offset)
    return float(max_search + 1)


def days_from_prev_holiday(date: pd.Timestamp, max_search: int = 30) -> float:
    """直近の休日日からの日数。見つからなければ max_search+1 を返す。"""
    for offset in range(1, max_search + 1):
        candidate = date - timedelta(days=offset)
        if not is_business_day(candidate):
            return float(offset)
    return float(max_search + 1)


def payday_window_flag(date: pd.Timestamp, window: int = 2) -> float:
    """月末±window 営業日内にあるかを判定するフラグ。"""
    if pd.isna(date):
        return 0.0
    month_end = (date + MonthEnd(0)).normalize()
    distance = business_days_distance(date, month_end)
    return float(1.0 if abs(distance) <= window else 0.0)


def fiscal_period_flag(date: pd.Timestamp) -> float:
    """四半期末（3,6,9,12月）を fiscal period としてフラグ化。"""
    if pd.isna(date):
        return 0.0
    return float(1.0 if date.month in (3, 6, 9, 12) else 0.0)


def trailing_run_length(values: List[float], window: int, condition) -> int:
    """末尾から window 件を対象に condition を満たす連続長を計算。"""
    count = 0
    for value in reversed(values[-window:]):
        if condition(value):
            count += 1
        else:
            break
    return count


def load_aggregated_data(base_dir: str) -> pd.DataFrame:
    """事前に集約済みの日次データを読み込む。"""
    path = os.path.join(base_dir, "prepared", AGGREGATED_INPUT_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aggregated input not found: {path}")
    df = pd.read_parquet(path)
    df["file_date"] = pd.to_datetime(df["file_date"])
    return df


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------------------
# 特徴量生成用の状態クラス
# --------------------------------------------------------------------------------------

@dataclass
class FeatureState:
    """
    逐次的にラグ・ローリング特徴量を更新するための状態クラス。

    推論時は予測値を `add_observation` で投入し続けることで、マルチステップ予測でも
    最新の状態が反映された特徴量を生成できる。
    """

    min_history: int = MIN_HISTORY
    max_history: int = MAX_HISTORY
    ewm_spans: Tuple[int, ...] = EWM_SPANS

    history_values: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    history_dates: Deque[pd.Timestamp] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    last_nonzero_date: Optional[pd.Timestamp] = None
    ewm_state: Dict[int, Optional[float]] = field(
        default_factory=lambda: {span: None for span in EWM_SPANS}
    )
    dow_counts: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(7)})
    dow_nonzero_counts: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(7)})
    bias_history: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    last_actual_value: Optional[float] = None

    def add_observation(self, date: pd.Timestamp, value: float, is_actual: bool = True) -> None:
        """最新観測（実績または予測）を状態に追加する。"""
        if pd.isna(date):
            return
        value = float(value)
        self.history_values.append(value)
        self.history_dates.append(date)
        if value > 0:
            self.last_nonzero_date = date

        weekday = int(date.weekday())
        if is_actual:
            self.dow_counts[weekday] = self.dow_counts.get(weekday, 0) + 1
            if value > 0:
                self.dow_nonzero_counts[weekday] = self.dow_nonzero_counts.get(weekday, 0) + 1

        for span in self.ewm_spans:
            prev = self.ewm_state.get(span)
            if prev is None or np.isnan(prev):
                self.ewm_state[span] = value
            else:
                alpha = 2.0 / (span + 1.0)
                self.ewm_state[span] = alpha * value + (1 - alpha) * prev

        if is_actual:
            self.last_actual_value = value
            ratio = 1.0
        else:
            denominator: Optional[float] = self.last_actual_value
            if (denominator is None or denominator == 0.0) and len(self.history_values) >= 2:
                denominator = self.history_values[-2]
            ratio = safe_ratio(value, denominator, default=1.0)
        self.bias_history.append(ratio)

    # ----------------------------------------------------------------------------------

    def _history_list(self) -> List[float]:
        return list(self.history_values)

    def _require_history(self) -> bool:
        return len(self.history_values) >= self.min_history

    def _calendar_features(self, date: pd.Timestamp) -> Dict[str, float]:
        weekday = date.weekday()
        return {
            "dow": float(weekday),
            "is_weekend": float(weekday >= 5),
            "is_business_day_f": float(is_business_day(date)),
            "month": float(date.month),
            "quarter": float(date.quarter),
            "day_of_month": float(date.day),
            "iso_week": float(date.isocalendar().week),
            "is_month_start": float(date.day <= 7),
            "is_month_end": float(date.day >= 24),
            "days_to_next_holiday_f": days_to_next_holiday(date),
            "days_from_prev_holiday_f": days_from_prev_holiday(date),
            "is_payday_window_f": payday_window_flag(date),
            "fiscal_period_f": fiscal_period_flag(date),
        }

    def build_feature_row(
        self,
        current_date: pd.Timestamp,
        static_info: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """
        現在の状態から特徴量行を生成する。

        Returns:
            特徴量辞書（FEATURE_COLUMNSの各キーを含む）もしくは None（履歴不足時）
        """
        if not self._require_history():
            return None

        history = self._history_list()
        history_dates = list(self.history_dates)
        features: Dict[str, float] = {}

        def last_value(offset: int) -> float:
            return float(history[-offset]) if len(history) >= offset else np.nan
        def tail(window: int) -> List[float]:
            return history[-window:] if len(history) >= window else history[:]

        # ラグ
        features["lag_1_f"] = last_value(1)
        features["lag_2_f"] = last_value(2)
        features["lag_3_f"] = last_value(3)
        features["lag_7_f"] = last_value(7)
        features["lag_14_f"] = last_value(14)
        features["lag_28_f"] = last_value(28)

        # ローリング
        for window in ROLLING_WINDOWS:
            window_values = tail(window)
            features[f"rolling_mean_{window}_f"] = float(np.mean(window_values))
            features[f"rolling_min_{window}_f"] = float(np.min(window_values))
            features[f"rolling_max_{window}_f"] = float(np.max(window_values))

        for window in STD_WINDOWS:
            window_values = tail(window)
            features[f"rolling_std_{window}_f"] = float(np.std(window_values, ddof=0))

        for window in NONZERO_WINDOWS:
            window_values = tail(window)
            nonzero_ratio = sum(1 for v in window_values if v > 0) / float(window) if window_values else 0.0
            features[f"recent_nonzero_ratio_{window}_f"] = float(nonzero_ratio)

        # 追加ローリング統計
        last_7 = tail(7)
        last_14 = tail(14)
        if last_7:
            features["rolling_median_7_f"] = float(np.median(last_7))
        else:
            features["rolling_median_7_f"] = 0.0
        if last_14:
            features["rolling_median_14_f"] = float(np.median(last_14))
            q75, q25 = np.percentile(last_14, [75, 25])
            features["rolling_iqr_14_f"] = float(q75 - q25)
            mean_14 = float(np.mean(last_14))
            std_14 = float(np.std(last_14, ddof=0))
            features["rolling_cv_14_f"] = safe_ratio(std_14, mean_14, default=0.0)
            features["peak_to_mean_ratio_14_f"] = safe_ratio(float(np.max(last_14)), mean_14, default=0.0)
        else:
            features["rolling_median_14_f"] = 0.0
            features["rolling_iqr_14_f"] = 0.0
            features["rolling_cv_14_f"] = 0.0
            features["peak_to_mean_ratio_14_f"] = 0.0

        # 連続日数関連
        features["recent_nonzero_run_length_7_f"] = float(trailing_run_length(history, 7, lambda v: v > 0))
        features["recent_nonzero_run_length_14_f"] = float(trailing_run_length(history, 14, lambda v: v > 0))
        features["zero_run_length_7_f"] = float(trailing_run_length(history, 7, lambda v: v <= 0))
        features["zero_run_length_14_f"] = float(trailing_run_length(history, 14, lambda v: v <= 0))

        # 週末需要の評価
        recent_weekend_values = [
            val
            for val, dt in zip(history, history_dates)
            if dt.weekday() >= 5 and (current_date - dt).days <= WEEKEND_LOOKBACK_DAYS and dt < current_date
        ]
        if recent_weekend_values:
            weekend_nonzero = sum(1 for v in recent_weekend_values if v > 0)
            weekend_rate = weekend_nonzero / float(len(recent_weekend_values))
            weekend_zero_flag = 1.0 if np.mean(recent_weekend_values) <= WEEKEND_ZERO_THRESHOLD else 0.0
        else:
            weekend_rate = 0.0
            weekend_zero_flag = 0.0
        features["weekend_nonzero_rate_f"] = float(weekend_rate)
        features["weekend_zero_flag_f"] = float(weekend_zero_flag)

        # 曜日別非ゼロ率
        for dow in range(7):
            total = self.dow_counts.get(dow, 0)
            nonzero = self.dow_nonzero_counts.get(dow, 0)
            features[f"dow_nonzero_rate_f_{dow}"] = float(nonzero / total) if total > 0 else 0.0

        # EWM
        for span in self.ewm_spans:
            features[f"ewm_mean_{span}_f"] = float(
                self.ewm_state.get(span) if self.ewm_state.get(span) is not None else 0.0
            )

        # 期間ギャップ
        if self.history_dates:
            last_date = self.history_dates[-1]
            features["days_since_last_obs_f"] = float((current_date - last_date).days)
            if self.last_nonzero_date is not None:
                features["days_since_last_nonzero_f"] = float(
                    (current_date - self.last_nonzero_date).days
                )
            else:
                features["days_since_last_nonzero_f"] = float((current_date - last_date).days)
            features["gap_business_days_f"] = float(count_business_days(last_date, current_date))
        else:
            features["days_since_last_obs_f"] = 0.0
            features["days_since_last_nonzero_f"] = 0.0
            features["gap_business_days_f"] = 0.0

        # 累積統計
        features["cumulative_mean_f"] = float(np.mean(history))
        features["cumulative_std_f"] = float(np.std(history, ddof=0))

        # トレンド
        for lag in TREND_LAGS:
            prev = last_value(lag)
            features[f"trend_{lag}_ratio_f"] = safe_ratio(history[-1], prev)
        last_28 = tail(28)
        if len(last_28) >= 5:
            x = np.arange(len(last_28))
            slope, _ = np.polyfit(x, last_28, deg=1)
            features["trend_28_slope_f"] = float(slope)
        else:
            features["trend_28_slope_f"] = 0.0

        # 月次モメンタム
        current_month_vals = [
            val
            for val, dt in zip(history, history_dates)
            if dt.year == current_date.year and dt.month == current_date.month and dt < current_date
        ]
        prev_month_anchor = current_date - DateOffset(months=1)
        prev_month_vals = [
            val
            for val, dt in zip(history, history_dates)
            if dt.year == prev_month_anchor.year and dt.month == prev_month_anchor.month and dt < current_date
        ]
        current_mean = float(np.mean(current_month_vals)) if current_month_vals else float(static_info.get("product_mean_f", 0.0))
        prev_mean = float(np.mean(prev_month_vals)) if prev_month_vals else float(static_info.get("product_mean_f", 0.0))
        features["month_over_month_ratio_f"] = safe_ratio(current_mean, prev_mean)

        # 季節要因
        month = int(current_date.month)
        month_mean_map: Dict[int, float] = static_info.get("month_mean_map", {})
        month_std_map: Dict[int, float] = static_info.get("month_std_map", {})
        seasonal_mean = month_mean_map.get(month, static_info.get("product_mean_f", 0.0))
        seasonal_std = month_std_map.get(month, static_info.get("product_std_f", 0.0))
        features["seasonal_month_mean_f"] = float(seasonal_mean)
        features["seasonal_month_std_f"] = float(seasonal_std)
        iso_week = int(current_date.isocalendar().week)
        week_mean_map: Dict[int, float] = static_info.get("week_mean_map", {})
        week_std_map: Dict[int, float] = static_info.get("week_std_map", {})
        week_mean = week_mean_map.get(iso_week, seasonal_mean)
        week_std = week_std_map.get(iso_week, seasonal_std)
        features["seasonal_week_mean_f"] = float(week_mean)
        features["seasonal_week_std_f"] = float(week_std)

        # バイアス指標
        if self.bias_history:
            bias_values = list(self.bias_history)
            features["cumulative_pred_to_actual_ratio_f"] = float(np.mean(bias_values))
            features["recent_bias_ratio_14_f"] = float(np.mean(bias_values[-14:]))
        else:
            features["cumulative_pred_to_actual_ratio_f"] = 1.0
            features["recent_bias_ratio_14_f"] = 1.0

        # 定数（静的）特徴量
        for key in STATIC_FEATURES:
            features[key] = float(static_info.get(key, 0.0))

        # カレンダ特徴量
        features.update(self._calendar_features(current_date))

        return features


# --------------------------------------------------------------------------------------
# 静的情報（商品プロファイル）の作成
# --------------------------------------------------------------------------------------

def build_material_index_map(material_keys: Iterable[str]) -> Dict[str, int]:
    """Material Key を 0 ベースの整数にマッピングする。"""
    return {key: idx for idx, key in enumerate(sorted(material_keys))}


def prepare_static_info(
    train_df: pd.DataFrame,
    material_index_map: Dict[str, int],
    volume_threshold: float = VOLUME_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """商品ごとの統計情報を作成する。"""
    stats = (
        train_df.groupby("material_key")["actual_value"]
        .agg(["mean", "std", "median", "min", "max", "sum", "count"])
        .rename(
            columns={
                "mean": "product_mean_f",
                "std": "product_std_f",
                "median": "product_median_f",
                "min": "product_min_f",
                "max": "product_max_f",
                "sum": "product_sum_f",
                "count": "product_count_f",
            }
        )
    )

    month_stats = train_df.copy()
    month_stats["month"] = month_stats["file_date"].dt.month
    month_stats = (
        month_stats.groupby(["material_key", "month"])["actual_value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "month_mean", "std": "month_std"})
    )

    week_stats = train_df.copy()
    iso_calendar = week_stats["file_date"].dt.isocalendar()
    week_stats["iso_week"] = iso_calendar.week.astype(int)
    week_stats = (
        week_stats.groupby(["material_key", "iso_week"])["actual_value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "week_mean", "std": "week_std"})
    )

    static_info: Dict[str, Dict[str, float]] = {}
    for material_key, idx in material_index_map.items():
        if material_key in stats.index:
            row = stats.loc[material_key]
        else:
            row = pd.Series(
                {
                    "product_mean_f": 0.0,
                    "product_std_f": 0.0,
                    "product_median_f": 0.0,
                    "product_min_f": 0.0,
                    "product_max_f": 0.0,
                    "product_sum_f": 0.0,
                    "product_count_f": 0.0,
                }
            )

        month_mean_map: Dict[int, float] = {}
        month_std_map: Dict[int, float] = {}
        week_mean_map: Dict[int, float] = {}
        week_std_map: Dict[int, float] = {}
        if material_key in month_stats.index.get_level_values(0):
            month_slice = month_stats.loc[material_key]
            if isinstance(month_slice, pd.Series):
                # 単月のみの場合 Series になるので dict へ変換
                month_mean_map[int(month_slice.name)] = float(month_slice["month_mean"])
                month_std_map[int(month_slice.name)] = float(
                    month_slice["month_std"] if not np.isnan(month_slice["month_std"]) else 0.0
                )
            else:
                for month, values in month_slice.iterrows():
                    month_mean_map[int(month)] = float(values["month_mean"])
                    month_std_map[int(month)] = float(
                        values["month_std"] if not np.isnan(values["month_std"]) else 0.0
                    )
        if material_key in week_stats.index.get_level_values(0):
            week_slice = week_stats.loc[material_key]
            if isinstance(week_slice, pd.Series):
                week_mean_map[int(week_slice.name)] = float(week_slice["week_mean"])
                week_std_map[int(week_slice.name)] = float(
                    week_slice["week_std"] if not np.isnan(week_slice["week_std"]) else 0.0
                )
            else:
                for week, values in week_slice.iterrows():
                    week_mean_map[int(week)] = float(values["week_mean"])
                    week_std_map[int(week)] = float(
                        values["week_std"] if not np.isnan(values["week_std"]) else 0.0
                    )

        volume_segment = 1.0 if row["product_sum_f"] >= volume_threshold else 0.0

        static_info[material_key] = {
            "material_idx": float(idx),
            "product_mean_f": float(row["product_mean_f"]) if not np.isnan(row["product_mean_f"]) else 0.0,
            "product_std_f": float(row["product_std_f"]) if not np.isnan(row["product_std_f"]) else 0.0,
            "product_median_f": float(row["product_median_f"]) if not np.isnan(row["product_median_f"]) else 0.0,
            "product_min_f": float(row["product_min_f"]) if not np.isnan(row["product_min_f"]) else 0.0,
            "product_max_f": float(row["product_max_f"]) if not np.isnan(row["product_max_f"]) else 0.0,
            "volume_segment_f": float(volume_segment),
            "month_mean_map": month_mean_map,
            "month_std_map": month_std_map,
            "week_mean_map": week_mean_map,
            "week_std_map": week_std_map,
        }

    return static_info


# --------------------------------------------------------------------------------------
# 特徴量生成メインロジック
# --------------------------------------------------------------------------------------

def generate_training_features(
    aggregated_df: pd.DataFrame,
    train_end_date: str,
    min_history: int = MIN_HISTORY,
    max_history: int = MAX_HISTORY,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    学習用の特徴量を生成する。

    Returns:
        features_df, static_info_map
    """
    train_end = pd.to_datetime(train_end_date)
    train_df = aggregated_df[aggregated_df["file_date"] <= train_end].copy()
    if train_df.empty:
        raise ValueError("No training data found up to train_end_date.")

    material_index_map = build_material_index_map(train_df["material_key"].unique())
    static_info_map = prepare_static_info(train_df, material_index_map)

    records: List[Dict[str, float]] = []

    for material_key, group in train_df.groupby("material_key"):
        group = group.sort_values("file_date")
        state = FeatureState(min_history=min_history, max_history=max_history)
        static_info = static_info_map[material_key]

        for row in group.itertuples(index=False):
            current_date: pd.Timestamp = row.file_date
            current_value: float = float(row.actual_value)

            feature_row = state.build_feature_row(current_date, static_info)
            if feature_row is not None:
                record = {
                    "material_key": material_key,
                    "file_date": current_date,
                    "actual_value": current_value,
                }
                for col in FEATURE_COLUMNS:
                    record[col] = feature_row[col]
                records.append(record)

            state.add_observation(current_date, current_value)

    if not records:
        raise RuntimeError("No feature rows were generated – consider reducing MIN_HISTORY.")

    features_df = pd.DataFrame(records)
    features_df.sort_values(["material_key", "file_date"], inplace=True)

    return features_df, static_info_map


def save_static_info(
    static_info_map: Dict[str, Dict[str, float]],
    output_dir: str,
) -> str:
    """静的情報を JSON として保存する。"""
    ensure_directory(output_dir)

    # JSON シリアライズ可能な形式へ変換（タイムスタンプなどは不要）
    serializable: Dict[str, Dict[str, object]] = {}
    for key, info in static_info_map.items():
        serializable[key] = {
            "material_idx": info["material_idx"],
            "product_mean_f": info["product_mean_f"],
            "product_std_f": info["product_std_f"],
            "product_median_f": info["product_median_f"],
            "product_min_f": info["product_min_f"],
            "product_max_f": info["product_max_f"],
            "volume_segment_f": info["volume_segment_f"],
            "month_mean_map": {str(k): float(v) for k, v in info["month_mean_map"].items()},
            "month_std_map": {str(k): float(v) for k, v in info["month_std_map"].items()},
            "week_mean_map": {str(k): float(v) for k, v in info.get("week_mean_map", {}).items()},
            "week_std_map": {str(k): float(v) for k, v in info.get("week_std_map", {}).items()},
        }

    path = os.path.join(output_dir, STATIC_INFO_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return path


def save_features(features_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    ensure_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(output_dir, FEATURE_FILENAME_LATEST)
    dated_path = os.path.join(output_dir, FEATURE_FILENAME_TEMPLATE.format(timestamp=timestamp))

    for path in (latest_path, dated_path):
        features_df.to_parquet(path, index=False)

    return latest_path, dated_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(base_dir: str = DEFAULT_BASE_DIR, train_end_date: str = "2024-12-31") -> None:
    print("=" * 60)
    print("Product-level Feature Generation (Sequential Ready)")
    print(f"Timestamp: {datetime.now()}")
    print(f"Parameters: base_dir={base_dir}, train_end_date={train_end_date}")
    print("=" * 60)

    aggregated_df = load_aggregated_data(base_dir)
    print(f"Loaded aggregated data: {len(aggregated_df):,} rows, "
          f"{aggregated_df['material_key'].nunique():,} material keys")
    print(f"Date range: {aggregated_df['file_date'].min()} ~ {aggregated_df['file_date'].max()}")

    features_df, static_info_map = generate_training_features(
        aggregated_df=aggregated_df,
        train_end_date=train_end_date,
        min_history=MIN_HISTORY,
        max_history=MAX_HISTORY,
    )

    print(f"\nGenerated training features: {len(features_df):,} rows, "
          f"{features_df['material_key'].nunique():,} material keys")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    features_dir = os.path.join(base_dir, "features")
    latest_path, dated_path = save_features(features_df, features_dir)
    static_path = save_static_info(static_info_map, features_dir)

    print("\nSaved files:")
    print(f"  Latest features : {latest_path}")
    print(f"  Timestamped     : {dated_path}")
    print(f"  Static info     : {static_path}")

    print("\nSummary statistics:")
    print(features_df["actual_value"].describe())

    print("\nFeature generation completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate product-level features with sequential support.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"データのベースディレクトリ (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="2024-12-31",
        help="学習データの終了日 (YYYY-MM-DD)",
    )

    args = parser.parse_args()
    main(base_dir=args.base_dir, train_end_date=args.train_end_date)
