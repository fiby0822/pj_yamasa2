# imp2 精度改善に向けた事前分析メモ

## 1. 分析セットアップ
- 分析スクリプト: `work/script/tmp/imp2/analyze_accuracy_comparison.py` (`202-224` 行で集計を実行)
- 入力データ:
  - 現行精度サマリ `work/vis_2025-01_to_03_imp2_2/summary/accuracy_summary.csv`
  - 予測詳細 (2025/1-3) `work/data/predictions/product_level_predictions_20251021_172800.parquet`
  - 学習期間の実績 (〜2024/12/31) `work/data/prepared/df_confirmed_order_input_yamasa_fill_zero.parquet`
- 出力（再分析時は `python3 work/script/tmp/imp2/analyze_accuracy_comparison.py` を実行）:
  - `work/script/tmp/imp2/out/key_feature_matrix.csv`
  - `work/script/tmp/imp2/out/group_comparison.csv`
  - `work/script/tmp/imp2/out/monthly_profile_high.csv` / `..._low.csv`

accuracy上位 25 件 (`accuracy >= 62.3%`) を「high」、下位 25 件 (`<= 42.5%`) を「low」と定義した。

## 2. グループ比較サマリ
`group_comparison.csv` より抜粋（平均値ベース）。

| 指標 | high | low | 差異メモ |
| --- | --- | --- | --- |
| 日次平均需要 (`daily_mean__mean`) | 543.2 | 60.3 | 高精度グループは需要規模が約9倍 |
| 日次変動係数 (`daily_cv__mean`) | 1.13 | 1.98 | 低精度はボラティリティが高い |
| 月次変動係数 (`monthly_cv__mean`) | 0.42 | 0.58 | 月次でも低精度群の方が変動大 |
| 最大/中央値比 (`seasonality_ratio__mean`) | 2.29 | 3.80 | 低精度群はピーク依存度が高い |
| 学習期非ゼロ日比率 (`nonzero_share__mean`) | 0.61 | 0.55 | 低精度群はゼロ日がやや多い |
| 予測期ゼロ日比率 (`test_zero_share__mean`) | 0.37 | 0.47 | テスト期でもゼロが多く予測難度↑ |
| 予測期変動係数 (`test_actual_cv__mean`) | 0.96 | 2.00 | 低精度群は実績のブレが2倍以上 |

### 考察
1. **需要規模と安定性が精度を左右**  
   高精度キーは大ロット・高頻度 (`daily_mean` 高、`daily_cv` 低) で学習サンプルが豊富。低精度キーは日次需要が少なく、曜日・季節によるスパイクで `seasonality_ratio` が悪化。

2. **ゼロ需要日の扱いが課題**  
   低精度群はゼロ日率が高く、`test_zero_share` も 47% と高い。ゼロ日の発生パターン学習や閾値制御の強化が必要。

3. **月次プロファイルの違い**  
   `monthly_profile_high.csv` と `...low.csv` では、高精度群が年間を通じて高水準で推移する一方、低精度群は 2024/9〜12 にかけて急伸と急落を繰り返し、直近月の情報だけではキャッチアップしにくい形状。

## 3. 個別キーの特徴
- **高精度代表 (A01120A)**: 需要安定 (`daily_cv=0.83`)、ゼロ日比率 36.7%、予測バイアスも小さい (`test_bias_mean=5.1`)。季節性は穏やか (`seasonality_ratio=1.38`)。
- **ボリュームは大きいが低精度 (A067414)**: 日次平均 603 だが `daily_cv=2.59` と激しく変動、ゼロ日 61%。週末ゼロ・突発需要が混在し、現状の特徴量ではピークタイミングを捉え切れていない。
- **超低ボリューム (A186710)**: 日次平均 1.3、ゼロ日率 72%。薄いデータでスパイクが稀に発生 (`seasonality_ratio=6.5`)。分類問題としてゼロ需要を判定後に数量を予測するアプローチが必要かもしれない。

## 4. 改善アイデアの方向性
1. **ゼロ日パターンのモデリング強化**  
   - 低精度群では週末・特定曜日にゼロが集中。ゼロ発生の事前確率推定（例: 2段階モデル or 週末専用特徴）を追加検討。
2. **スパイク検知と短期トレンド特徴の補強**  
   - `seasonality_ratio` が高いキー向けに、直近 14〜28 日以内のピーク位置、営業日カウントなどをより解像度高く持たせる。
3. **低ボリュームSKUの扱い分岐**  
   - `daily_mean < 20` 等のカテゴリを設け、XGBoost とは別のヒューリスティクス（直近移動平均、sMAPE最小の単純モデル）とアンサンブルする。
4. **需要急伸キーの検証**  
   - A067414 など高ボリュームだが精度低いSKUは個別に時系列をチェックし、外部要因やプロモーション日付の有無をヒアリング。必要なら特徴量にイベントフラグを加える。

## 5. 次ステップの提案
1. `key_feature_matrix.csv` を元に閾値（例: `seasonality_ratio > 3` かつ `daily_mean < 50`）で改善対象リストを抽出し、優先順位付け。
2. 低精度グループに対して週末ゼロ判定モデル・スパイク検出ロジックのPoCを `work/script/tmp/imp2` 配下で試作。
3. 改善案を反映した小規模な再学習を実施し、`work/vis_...` で再評価する流れを準備。

※追加の可視化・再集計は `work/script/tmp/imp2` 以下でのみ実施してください。***
