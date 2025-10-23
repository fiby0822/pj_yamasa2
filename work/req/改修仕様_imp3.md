# imp3 特徴量強化仕様（検討版）

## 0. 目的と前提
- 対象: `work/script/generate_features_product_level.py` および付随する特徴量保存処理。
- ゴール: `work/vis_2025-01_to_03_imp2_2` 時点の精度課題を、「特徴量の拡張・精緻化」のみで改善するための改修案を整備する。
- 参考分析: `work/req/imp2_accuracy_pre_analysis.md` ならびに `work/script/tmp/imp2/out/` の集計結果  
  - 低精度群の特徴: 日次CV 1.98、月次CV 0.58、季節ピーク比 3.8、テスト期ゼロ日率 47% と高ボラティリティ/ゼロ集中。
  - 高精度群: 日次CV 1.13、ゼロ日率 36%、ピーク比 2.29 と安定。

本仕様では「特徴量開発」に焦点を当て、モデル学習・推論ロジックへは極力手を入れずに改善インパクトを狙う。

## 1. 特徴量設計の指針
1. 需要ボラティリティのセグメント化を明示し、学習時に扱いやすい静的指標を充実させる。
2. ゼロ需要発生のパターンを特徴量として提供し、モデルが自律的にゼロ判定を学習できるようにする。
3. スパイク発生やトレンド転換を検知する派生指標を追加し、短期変化の検出力を高める。
4. 既存特徴量との冗長性を避けるため、算出ロジックと欠損処理を整理しつつ FeatureState で再利用できる形に統合する。

## 2. 追加・拡張する特徴量
### 2-1. 静的セグメント指標
- `generate_features_product_level.py::prepare_static_info`
  - 既存の集計値から以下を導出し、静的情報に追加する。FeatureState でも参照可能とする。
    - `segment_high_volume_f`: `daily_mean >= 200` または `total_volume >= 30,000`
    - `segment_low_volume_f`: `daily_mean < 20` または `nonzero_share < 0.4`
    - `segment_spiky_f`: `seasonality_ratio >= 3.0` または `daily_cv >= 2.0`
  - 付随して `volume_segment_f` の閾値を最新分布に合わせて再検証（既定: 5,000 → 要確認）。

### 2-2. ゼロ需要検知向け特徴
- `FeatureState.build_feature_row` で下記の算出を行い、日次特徴として保持する。
  - `weekday_zero_prob_f_{0..6}`: 直近12週間の曜日別ゼロ出現率（ハドリング調整含む）。
  - `zero_run_probability_f`: `zero_run_length_7_f` をシグモイド変換したゼロ連続確率。
  - `recent_nonzero_gap_f`: 直近非ゼロ実績までの日数を対数スケールで変換した指標。
  - 欠損値は 0.0 で補完し、静的情報にゼロ発生統計（`weekday_zero_share_f_*` など）を併記する。

### 2-3. スパイク・トレンド派生特徴
- `FeatureState` に以下の派生量を追加。
  - `recent_peak_position_28_f`: 過去28日のピークからの経過日数。
  - `recent_growth_ratio_7_28_f`: `rolling_mean_7_f / max(rolling_mean_28_f, eps)`。
  - `weekday_shift_score_f`: 直近4週の曜日別平均と半年平均の差を標準化した Z スコア。
  - `momentum_decay_ratio_f`: `ewm_mean_7_f` と `ewm_mean_30_f` の差分比率で、中期トレンドへの回帰度合いを測る。

### 2-4. エッジケース対応
- 製品によっては履歴不足で FeatureState 出力が欠損する問題があるため、以下の対策を行う。
  - 7日未満の場合は最短ウィンドウでの計算値を使用し、`feature_missing_flag_*` を新設して欠損状態をモデルへ渡す。
  - 長期欠損が続く SKU を静的にマークする `long_gap_flag_f` を追加し、学習時の識別に用いる。

## 3. 実装影響範囲
- 主に `generate_features_product_level.py` のみを改修対象とし、出力パーケットおよび静的情報のスキーマ変更に合わせてドキュメントを更新する。
- `FeatureState` クラスの初期化ロジック、`prepare_static_info`、`generate_training_features`（レコード生成部）の修正が中心。
- XGBoost 学習スクリプトや可視化ロジックは利用側として追加列を自動取り込みするのみで、直接のコード変更は想定しない。

## 4. 検証計画
1. 特徴量検証: 新規列が学習・推論双方で欠損なく出力されることを単体テスト（小規模データセット）で確認。
2. オフライン比較: `imp2` 学習済みモデルに新列を追加投入し、2025/01-03 期間で feature importance と精度変化を評価。
3. 回帰確認: 代表SKU（A01120A, A030830 など）で特徴量プロファイルが合理的に更新されているかを可視化で確認。

## 5. 想定工数とリスク
- 実装工数: 2〜3日（FeatureState 拡張・静的情報の整備・単体検証を含む）。
- 主なリスク:
  - 特徴量の増加によりモデルが過学習する可能性 → 重要度分析と早期ストッピング設定でモニタ。
  - 既存パーケット互換性の破壊 → スキーマ変更を伴う場合はバージョン管理を導入（例: `features_v3` ディレクトリ）。

---
本仕様は imp3 における特徴量開発のたたき台であり、実装時に算出コスト・冗長性を踏まえて優先度を精査する。***
