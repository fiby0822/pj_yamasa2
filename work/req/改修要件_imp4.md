# imp3 精度向上仕様案（検討版）

## 0. 目的と前提
- 対象: `work/script` 配下の学習・推論パイプライン（特徴量生成・モデル学習・逐次推論ロジック）。
- 目的: `work/vis_2025-01_to_03_imp3` の精度を向上させ、平均 Accuracy 55% / WAPE 45% / 予測総量比 ±5% 以内を目標とする。
- 参考資料: `work/req/精度レポート_imp3.md`, `work/script/tmp/imp3/out/` の分析成果物。
- 現状課題:
  - 高ボリューム SKU の過大予測（予測/実績 1.2～1.4 倍、Accuracy <55% の SKU が 26 件）。
  - 低ボリューム/スパイク SKU におけるゼロ日判定の誤検知と高 WAPE（92%）。
  - 月次で恒常的な過大予測（3 月で 1.25 倍）。

## 1. 改修対象の概要
1. ゼロ需要確率と縮小ロジックの再設計（高ボリューム/低ボリューム双方）。
2. スパイク・高ボラティリティ SKU 向けのピーク調整強化。
3. 月次季節性の再スケーリングと学習時の重み補正。

以下、各領域の仕様詳細を示す。

## 2. 高ボリューム SKU の過大予測抑制
### 2-1. ゼロ確率上限のセグメント別制御
- 現状: `weekday_zero_prob_f_*` は最大 0.6。高ボリューム SKU でもゼロ確率が過大になり、週末ゼロ強制で予測がゼロ寄り → 連日挽回しようと大きな値を出力し過大になる。
- 仕様:
  - `segment_high_volume_f >= 0.5` の SKU について、`weekday_zero_prob_f_*` を `min(raw, 0.3)` にクリップ。
  - `prob_zero` 合成時（推論側）も同様に 0.3 上限を適用し、ゼロ強制条件を `prob_zero >= 0.8` に引き上げる。
  - `pred_ratio` が 1.2 を超えている SKU は対象一覧に記録し、ログ出力へ追加（分析継続用）。

### 2-2. 逐次推論時の数量縮小
- 仕様:
  - `segment_high_volume_f` かつ `pred_ratio` が履歴上 1.1 以上の SKU では、予測値を `recent_actual_cumsum_28_f / 28` と `rolling_mean_14_f` の加重平均 (0.5/0.5) で制限。
  - `momentum_decay_ratio_f` が正（瞬間上昇）かつ過去 3 日で 30% 以上乖離した場合、過去値との比率上限 (1.15 倍) を設ける。

### 2-3. 学習時重み付け
- `segment_high_volume_f` かつ `pred_ratio > 1.2` の SKU に対して、学習サンプルで `recent_zero_share_28_f < 0.2` のレコード重みを 0.9、`recent_zero_share_28_f >= 0.2` のレコード重みを 1.1 に調整。  
  → ゼロ日付近と非ゼロのバランスを改善し過大推定を抑制。

## 3. 低ボリューム／長期ゼロ SKU 向け改修
### 3-1. 二段階予測の導入（ヒューリスティック）
- `segment_low_volume_f >= 0.5` または `long_gap_flag_f >= 0.5` の SKU を対象に以下の処理を追加:
  - (1) ゼロ/非ゼロ判定: `prob_zero` が 0.6 以上の場合は予測値を 0 に設定。
  - (2) 非ゼロ日の数量: `rolling_mean_14_f`, `rolling_median_14_f`, `recent_peak_actual_f` の最小値を初期予測とし、XGBoost 出力との重み付け (XGB 0.4 / ヒューリ 0.6) で合成。
- これにより極端な山型出力を抑制しつつ、ゼロ日判定を反映。

### 3-2. 特徴量の追加調整
- `recent_nonzero_gap_f` が高いケースでの減衰指標として `gap_decay_factor = exp(-recent_nonzero_gap_f)` を FeatureState に追加し、モデル入力に含める。
- `longest_zero_run` を静的特徴として渡し、学習時にゼロ傾向を保持。

## 4. スパイク・ボラティリティ対策
### 4-1. ピーク位置と増幅のクリップ
- `segment_spiky_f >= 0.5` に対して、`recent_peak_position_28_f <= 7` のとき `recent_peak_actual_f` を上限 1.1 倍でハードクリップ。
- `recent_growth_ratio_7_28_f > 1.4` かつ `daily_cv > 1.5` の場合、予測値を `momentum_decay_ratio_f` の符号に応じて ±15% 補正する（正のとき削減、負のとき維持）。

### 4-2. 異常スパイクのログ出力
- 推論時に `predicted > recent_peak_actual_f * 1.25` を検知したら SKU・日付・特徴量主要値を `tmp/imp3/logs/spike_predictions.csv` に追記し、改善効果を検証できる仕組みを整備。

## 5. 月次スケール調整
### 5-1. 季節性補正式
- `monthly_overview.csv` から得た過大比に基づき、月次での補正係数 `month_scale` を導入。
  - 2025 年 1,2,3 月はそれぞれ {0.82, 0.86, 0.80} を初期値とし、`seasonal_month_mean_f` の計算時に掛け合わせる。
  - 補正は学習/推論の両方で適用し、シーズンパターンを下方シフト。

### 5-2. モデル出力後の月次リスケール
- 推論完了後、各 SKU × 月の総量が実績の 1.05 倍を超える場合には `pred_ratio` 基づく線形縮小を適用（上限 5%）。  
  - この処理結果を `work/data/output/product_level_predictions_latest.csv` に反映し、履歴を `product_level_predictions_<timestamp>_scaled.parquet` として保存。

## 6. 実装メモ
- 主要改修ファイル: `generate_features_product_level.py`, `train_model_product_level.py`, `visualize_predictions.py`（ログ・検証補助）。  
- 既存 imp3 用の可視化・レポーティングは仕様確認用に更新するが、外部スクリプト（`work/script/tmp/imp3`）は分析用のみ変更。
- 実装順:  
  1. FeatureState/静的特徴の追加 (`gap_decay_factor`, `longest_zero_run`)  
  2. ゼロ確率クリップと二段階予測の導入  
  3. ピーク抑制・月次補正式の追加  
  4. ログ出力・再学習・検証

## 7. 成果検証
1. `work/vis_2025-01_to_03_imp3` を更新し、指標改善（Accuracy 55%以上、WAPE 45%以下、pred/actual 1.05 以内）を確認。
2. `segment_high_volume_f` と `segment_low_volume_f` 各グループの Accuracy/WAPE を比較し改善幅を評価。
3. `monthly_overview.csv` を再出力し、月別 pred_ratio が ±5% に収まること。
4. 改修後のログ (`spike_predictions.csv`, `overpredict_log.csv`) を確認し、異常大量出力が減少しているかを手動確認。

## 8. リスクと留意点
- ゼロ確率の強い抑制により実際にゼロが多い SKU で過小化する恐れ → PoC 期間中は `tmp/imp3` 内で差分チェックログを保持する。
- 月次補正式は 2025/01-03 の過大を基準としているため、他期間への適用時には再推定が必要。
- モデル外のリスケールを導入するため、予測値の整合性を維持するロジック（再学習時の再現）をドキュメント化する。

---
本仕様は imp3 時点の課題解決を目的とした改修案であり、実装後に効果測定 → ノウハウ取り込み → 次フェーズ（imp4）へ繋げる計画とする。***
