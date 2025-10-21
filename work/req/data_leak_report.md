# データリーク検査レポート

## 検査実施日時
2024年10月21日

## 【重大な問題】データリークが発見されました

### 1. リークの概要
- **問題**: 学習データに未来のデータ（2025年1月〜6月）が含まれています
- **影響**: モデルが未来の情報を使って学習しているため、予測精度が不当に高くなっている可能性があります

### 2. 発見されたリークの詳細

#### 2.1 データファイルの期間
| ファイル | 期間 | 問題 |
|---------|------|------|
| work/data/input/df_confirmed_order_input_yamasa.parquet | 2021-01-05 〜 **2025-06-30** | ⚠️ 2025年のデータが含まれている |
| work/data/input/df_confirmed_order_input_yamasa_fill_zero.parquet | 2021-01-05 〜 **2025-06-30** | ⚠️ 2025年のデータが含まれている |
| work/data/features/product_level_features_latest.parquet | 2021-01-05 〜 **2025-06-30** | ⚠️ 2025年のデータが含まれている |

- **2024年12月31日以降のレコード数**: 17,695件
- **リークしているデータの期間**: 2025年1月1日 〜 2025年6月30日

#### 2.2 根本原因
**`work/script/prepare_product_level_data_local.py`** の実装に問題があります：
- このスクリプトは`train_end_date`パラメータを受け取りますが、**実際にはデータのフィルタリングに使用していません**
- 結果として、`input`ディレクトリの全期間データ（〜2025年6月）がそのまま保存されています

#### 2.3 実行フローの問題
`scripts/run_product_level_pipeline.py`の実行フロー：
1. **Data Preparation** (`prepare_product_level_data_local.py`) - ⚠️ **train_end_dateパラメータを渡していない**
2. Feature Generation (`generate_features_product_level.py`) - train_end_dateで分割しようとするが、すでにデータが汚染されている
3. Model Training & Prediction

### 3. リーク防止メカニズムの状態

#### 3.1 実装されている対策（部分的に機能）
`generate_features_product_level.py`には以下のリーク防止機能が実装されています：
- `split_train_test()`関数で`train_end_date`に基づいてdata_typeを設定
- ラグ特徴量生成時に、testデータには学習期間のデータのみを使用
- 移動統計特徴量も同様の処理
- 商品プロファイル特徴量は学習データのみから計算

**しかし**、これらの対策は入力データ自体に未来のデータが含まれているため、完全には機能していません。

### 4. 推奨される修正

#### 4.1 緊急対応（データの再生成）
1. `prepare_product_level_data_local.py`を修正して、train_end_dateでデータをフィルタリング
2. すべての中間データ（prepared/、features/）を削除
3. パイプラインを最初から再実行

#### 4.2 コード修正案
`prepare_product_level_data_local.py`の修正が必要です：
```python
# 現在のコード（問題あり）
df = pd.read_parquet(input_file)  # 全期間を読み込み

# 修正案
df = pd.read_parquet(input_file)
df['file_date'] = pd.to_datetime(df['file_date'])
df = df[df['file_date'] <= train_end_date]  # train_end_date以前のデータのみ使用
```

### 5. 影響評価

現在のモデルは以下の理由で信頼できません：
1. 学習データに2025年1月〜6月のデータが含まれている
2. テスト期間（2025年1月）のデータで学習している可能性
3. 報告されている精度（92.69%）は過大評価されている可能性が高い

### 6. 結論

**重大なデータリークが確認されました。**

モデルの予測精度は実際より高く見積もられている可能性があり、本番環境での使用は推奨されません。早急にデータ準備スクリプトを修正し、モデルを再学習する必要があります。

---
*注: モジュールの改修は行っていません（要請に従い、確認のみ実施）*