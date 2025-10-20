# ヤマサ確定注文需要予測システム

## 概要
このシステムは、S3に格納されたExcelデータを使用して確定注文の需要予測を行います。
時系列データの特徴量生成、XGBoostによる機械学習モデルの構築、Walk-forward validationによる評価を実装しています。

## セットアップ

### 1. 仮想環境の有効化
```bash
source venv/bin/activate
```

### 2. 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### 必要なパッケージ
- boto3 (S3アクセス)
- pandas, numpy (データ処理)
- xgboost (機械学習モデル)
- optuna (ハイパーパラメータ最適化)
- scikit-learn (評価指標)
- matplotlib, seaborn (可視化)
- pyarrow (Parquetファイル処理)

## ディレクトリ構成
```
yamasa/
├── modules/                # コアモジュール
│   ├── core/              # データ準備のコアモジュール
│   │   ├── prepare/       # データ準備処理
│   │   └── archive/       # 過去のファイル
│   ├── config/            # 設定ファイル
│   │   └── feature_window_config.py  # 特徴量ウィンドウサイズ設定
│   ├── features/          # 特徴量生成モジュール
│   │   ├── timeseries_features.py    # 時系列特徴量生成
│   │   └── feature_generator_with_s3.py  # S3連携特徴量生成
│   ├── models/            # 機械学習モジュール
│   │   ├── train_predict.py  # モデル学習・予測
│   │   └── predictor.py      # 予測実行
│   ├── evaluation/        # 評価モジュール
│   │   └── metrics.py        # 評価指標計算
│   ├── data_io/          # データI/O
│   │   └── s3_handler.py     # S3ハンドラー
│   └── metrics/          # モデル比較・評価
├── scripts/              # 実行スクリプト
│   ├── prepare/         # データ準備スクリプト
│   │   ├── run_prepare_yamasa.py      # Excelファイル加工
│   │   └── run_prepare_fill_zero.py   # ゼロ値補完
│   ├── features/        # 特徴量生成スクリプト
│   │   ├── generate_features.py           # 特徴量生成メイン（統合版）
│   │   ├── generate_features_batch_s3.py  # バッチ処理（S3中間保存）
│   │   └── generate_features_by_usage_type.py  # usage_type別処理
│   ├── train/          # モデル学習スクリプト
│   │   └── train_model.py            # モデル学習メイン（統合版）
│   └── predict/        # 予測スクリプト
│       └── run_prediction.py         # 予測実行
├── notebooks/          # Jupyter Notebook格納
├── tests/             # テストスクリプト
├── logs/              # ログファイル
├── requirements.txt   # 依存パッケージ
├── run_prepare.sh     # データ準備実行シェル
└── run_train_predict.sh  # 学習・予測実行シェル
```

## 処理概要

### 1. データの取得・加工 (prepare_data_**.py)
1-1. S3に置かれているエクセルファイルを加工・結合し、S3に保存する
   - 入力: `s3://fiby-yamasa-prediction/input_data/*.xlsx`
   - 出力ファイル名: `df_confirmed_order_input_yamasa.parquet`

1-2. 1-1のデータはactual_value(実績値)が発生しているレコードしかないため、実績値がゼロのレコードを保存する必要がある
   - material_key毎にfile_dateのminとmaxを取得し、その間の期間に対して、actual_value=0のレコードを挿入する
   - 他の項目はfile_dateから遡って最新のレコードを補完する
   - 出力ファイル名: `df_confirmed_order_input_yamasa_fill_zero.parquet`

### 2. 特徴量の生成 (generate_features.py)
1のデータをもとに特徴量を生成する
- 特徴量はラグ特徴量やmaterial_key毎の週平均など、過去データを用いるものが多いが、train_end_dateで指定した期間の特徴量を用いる
- 即ち、train_end_dateより後の期間のactual_valueは欠損値として特徴量を全期間に対して計算する
- **デフォルトはusage_type別での特徴量生成**（business/household別に処理）

#### 🔴 特徴量命名規則（必須）
**すべての特徴量カラムは末尾に `_f` を付ける必要があります。**
- 正しい例: `lag_1_f`, `month_f`, `container_f`
- 誤った例: `lag_1`, `month`, `container`

⚠️ この規則は厳守すること。`_f` が付いていないカラムは特徴量として認識されません。

- 生成される特徴量:
  - ラグ特徴量（1, 2, 3期前）
  - 移動平均（2, 3, 6期間）
  - 移動標準偏差（2, 3, 6期間）
  - 累積平均（2, 3, 6, 12期間）
  - 変化率
  - 曜日、週番号、月、年の特徴量
  - container特徴量（usage_typeに基づく）

#### 出力ファイル:
- **usage_type別（デフォルト）**:
  - `data/features/confirmed_order_demand_yamasa_features_business_latest.parquet`
  - `data/features/confirmed_order_demand_yamasa_features_household_latest.parquet`
- **全データ版**:
  - `output/features/confirmed_order_demand_yamasa_features_latest.parquet`

### 3. 予測モデルの構築・予測の実行 (train_model.py)
- 2で生成した特徴量を用いる
- step_countで指定した月の数だけ予測を行う
  - 例: step_count=6, train_end_date=2024/12/31の場合、2025/1~2025/6までの6ヶ月分の予測結果を返す
- 予測対象のデータに対して、実績値がゼロより大きい値に対して予測する
- モデルは指示がなければxgboostを使い、ランダムサンプリングは行わない
- Walk-forward validation（月単位）で検証
- 外れ値処理: Winsorization、Hampelフィルタを適用（オプション）
- **デフォルトはusage_type別での学習・予測**

#### Material Keyフィルタリング:
- **Business**:
  - 学習期間: Top 3,000 material keys
  - テスト期間: 最低 step_count * 2 件以上
- **Household**:
  - 学習期間: Top 3,500 material keys
  - テスト期間: 最低 step_count * 4 件以上
- **全データ**:
  - 学習期間: Top 7,000 material keys
  - テスト期間: 最低 step_count * 4 件以上

### 4. 評価指標を算出し、表示・保存 (analyze_**.py)
モデル実行と同じ処理内で、下記項目を計算し表示・保存する。モデルや特徴量の変更があった場合は、変更前と変更後の値を表示する。

誤差率平均の定義: 「(予測値-実績値の絶対値)/実績値」をmaterial_key毎に平均する

- **基本評価指標**:
  - RMSE
  - MAE
  - 誤差率平均の平均値
  - 誤差率平均の中央値

- **誤差率平均分析**:
  - 誤差率平均が20%以内のmaterial_key数・割合
  - 誤差率平均が30%以内のmaterial_key数・割合
  - 誤差率平均が50%以内のmaterial_key数・割合

- **追加指標**:
  - MAPE（平均絶対パーセント誤差）
  - 相関係数
  - R²スコア

- **Material Key別評価**: 各material_keyの予測精度を個別に評価
- **可視化**: 実績vs予測散布図、誤差分布、時系列グラフ

## 🔴 重要：実行時の必須要件

### モデル学習・予測を実行する際は、以下の設定を厳守してください：

| パラメータ | デフォルト値 | 説明 |
|----------|------------|------|
| **train_end_date** | **2024-12-31** | 学習データの終了日（変更しない限りこの値を使用） |
| **step_count** | **1** | 予測月数（変更しない限り1ヶ月を使用） |
| **テスト期間** | **2025年1月** | train_end_date=2024-12-31の場合のテスト期間 |

⚠️ **注意**: 指示がない限り、必ず上記のデフォルト設定で実行してください。

### 正しい実行例：
```bash
python scripts/train/train_model.py --train-end-date "2024-12-31" --step-count 1
```

## 実行方法

### 推奨ワークフロー（usage_type別処理）

```bash
# 1. データ準備
./run_prepare.sh

# 2. 特徴量生成（usage_type別）
python3 scripts/features/generate_features.py --mode by_usage_type

# 3. モデル学習・予測（usage_type別）
python3 scripts/train/train_model.py --mode by_usage_type --train-end-date "2024-12-31" --step-count 1
```

### 方法1: 全体パイプラインの実行

#### データ準備
```bash
./run_prepare.sh
```

#### モデル学習・予測（基本）
```bash
./run_train_predict.sh train
```

#### モデル学習・予測（Optuna最適化付き）
```bash
./run_train_predict.sh train true
```

#### フルパイプライン（準備＋学習＋予測）
```bash
./run_train_predict.sh full
```

### 方法2: 個別実行

#### データ準備
```bash
# Excelファイルの加工
python3 scripts/prepare/run_prepare_yamasa.py

# ゼロ値補完
python3 scripts/prepare/run_prepare_fill_zero.py
```

#### 特徴量生成
```bash
# usage_type別に特徴量生成（デフォルト・推奨）
python3 scripts/features/generate_features.py --mode by_usage_type

# 全データで特徴量生成
python3 scripts/features/generate_features.py --mode all

# バッチ処理（メモリ制限がある場合）
python3 scripts/features/generate_features_batch_s3.py
```

#### モデル学習
```bash
# usage_type別で学習（デフォルト・推奨）
python3 scripts/train/train_model.py \
    --mode by_usage_type \
    --train-end-date "2024-12-31" \
    --step-count 1

# 全データで学習（従来版）
python3 scripts/train/train_model.py \
    --mode all \
    --train-end-date "2024-12-31" \
    --step-count 1

# 外れ値処理を有効化（処理時間増加）
python3 scripts/train/train_model.py \
    --mode by_usage_type \
    --train-end-date "2024-12-31" \
    --step-count 1 \
    --enable-outlier-handling

# GPU使用（利用可能な場合）
python3 scripts/train/train_model.py \
    --mode by_usage_type \
    --train-end-date "2024-12-31" \
    --step-count 6 \
    --use-gpu

# Optuna最適化付き学習（6ヶ月予測）
python3 scripts/train/train_model.py \
    --mode by_usage_type \
    --train-end-date "2024-12-31" \
    --step-count 6 \
    --use-optuna \
    --n-trials 50
```

#### 予測実行
```bash
# 将来予測
python3 scripts/predict/run_prediction.py \
    --mode future \
    --start-date "2025-01-01" \
    --end-date "2025-01-31" \
    --save-results
```

## S3構造
```
s3://fiby-yamasa-prediction/
├── input_data/                    # 入力Excelファイル
│   └── *.xlsx
└── output/                        # 処理結果
    ├── df_confirmed_order_input_yamasa.parquet           # 加工済みデータ
    ├── df_confirmed_order_input_yamasa_fill_zero.parquet # ゼロ値補完済み
    ├── features/                  # 特徴量データ
    │   ├── confirmed_order_demand_yamasa_features_latest.parquet  # 全データ版
    │   ├── confirmed_order_demand_yamasa_features_business_latest.parquet
    │   ├── confirmed_order_demand_yamasa_features_household_latest.parquet
    │   ├── confirmed_order_demand_yamasa_features_[timestamp].parquet
    │   └── temp_batches/          # バッチ処理の中間ファイル
    ├── models/                    # 学習済みモデル
    │   ├── confirmed_order_demand_yamasa_model_latest.pkl  # 全データ版
    │   ├── confirmed_order_demand_yamasa_model_business_latest.pkl
    │   ├── confirmed_order_demand_yamasa_model_household_latest.pkl
    │   ├── confirmed_order_demand_yamasa_params_latest.pkl
    │   └── confirmed_order_demand_yamasa_model_[timestamp].pkl
    ├── predictions/               # 予測結果
    │   ├── confirmed_order_demand_yamasa_predictions_latest.parquet  # 全データ版
    │   ├── confirmed_order_demand_yamasa_predictions_by_usage_type_latest.parquet
    │   └── confirmed_order_demand_yamasa_predictions_[timestamp].parquet
    └── evaluation/                # 評価結果
        ├── confirmed_order_demand_yamasa_metrics_latest.json
        ├── confirmed_order_demand_yamasa_predictions_latest.csv
        ├── confirmed_order_demand_yamasa_material_summary_latest.csv
        └── *.png                  # 可視化画像
```

### ローカル構造（usage_type別特徴量）
```
/home/ubuntu/yamasa/data/features/
├── confirmed_order_demand_yamasa_features_business_latest.parquet
└── confirmed_order_demand_yamasa_features_household_latest.parquet
```

## パラメータ説明

### feature_window_config.py
```python
WINDOW_SIZE_CONFIG = {
    "material_key": {
        "lag": [1, 2, 3],           # ラグ期間
        "rolling_mean": [2, 3, 6],  # 移動平均ウィンドウ
        "rolling_std": [2, 3, 6],   # 移動標準偏差ウィンドウ
        "cumulative_mean": [2, 3, 6, 12],  # 累積平均期間
    },
    # store_code, usage_type等も同様に設定
}
```

### generate_features.py
- `--mode`: 生成モード
  - `by_usage_type`（デフォルト）: usage_type別に特徴量生成
  - `all`: 全データで特徴量生成
- `--train-end-date`: 学習データの終了日（デフォルト: 2024-12-31）

### train_model.py
- `--mode`: 学習モード
  - `by_usage_type`（デフォルト）: usage_type別にモデル学習
  - `all`: 全データでモデル学習
- `--train-end-date`: 学習データの終了日（デフォルト: 2024-12-31）
- `--step-count`: 予測月数（デフォルト: 1）
- `--use-optuna`: Optunaでのハイパーパラメータ最適化
- `--n-trials`: Optunaの試行回数（デフォルト: 50）
- `--enable-outlier-handling`: 外れ値処理を有効化（デフォルト: 無効）
- `--use-gpu`: GPU使用（利用可能な場合）

### run_prediction.py
- `--mode`: 予測モード（future, walk-forward, material-key, single-date）
- `--start-date`, `--end-date`: 予測期間
- `--material-keys`: 対象Material Keyリスト
- `--aggregate`: 集約方法（sum, mean, median）
- `--save-results`: 結果をS3に保存

## 性能・制限事項

### メモリ要件
- 全データ処理（フィルタリング前）: 32GB以上必要
- **フィルタリング適用時: 8GB程度で動作可能**
- データサイズ: 約3,200万レコード、38,512 material keys
- **フィルタリング後: 約400万レコード、3,000-4,000 material keys**

### 処理時間（r6a.xlarge: 32GB RAM, 4 vCPU）
- データ準備: 約5-10分
- 特徴量生成: 約30-60分（バッチ処理）
- モデル学習:
  - **フィルタリング適用時: 約2-5分**
  - フィルタリング無し: 約10-30分

### データフィルタリング機能（v2024.10.15追加）
学習効率化のため、以下の条件でMaterial Keyを自動フィルタリング：
- **上位3000個のMaterial Key**（取引量ベース、約80%のカバー率）
- **テスト期間でアクティブなMaterial Key**（actual_value>0が一定数以上）

これにより：
- データ量を**約90%削減**
- メモリ使用量を**24.6GB→3GB**に削減
- 学習時間を**約1/10に短縮**

### 主要な機能
- ✅ 月単位のWalk-forward validation
- ✅ テストデータのみでの評価（過学習防止）
- ✅ train_end_dateによるデータリーク防止
- ✅ Material Key別の詳細評価
- ✅ S3との完全統合（ローカル保存なし）
- ✅ メモリ効率的なバッチ処理
- ✅ Optuna によるハイパーパラメータ最適化

## 注意事項
- AWS認証情報は環境変数またはIAMロールで管理
- S3へのアクセスにはap-northeast-1リージョンを使用
- ローカルへの保存は行わず、全てS3に保存
- データ量が大きいため、処理実行前に必ずリソースを確認

## Claude Codeにおける注意点
- データ量が非常に大きいため、「処理を実行してください」と明示的に指示されない限りコマンドは実行しない
- S3に新しくデータを出力する際は出力先パスを必ず表示する
- 新しくPythonファイルを作る場合はファイル名を表示し、処理概要をREADMEに追記する
- メモリ不足の場合はバッチ処理モードを推奨する
- **重要**: サンプリングによる実行及び実装は一切行わない。全データでの処理が必要な場合は、メモリ増設やバッチ処理などの他の解決策を提案する

## 学習完了後の処理

### EC2インスタンスの自動停止
学習・予測処理の完了後、コスト削減のためEC2インスタンスを自動停止できます：

```bash
# 学習実行と自動停止を組み合わせる
cd /home/ubuntu/yamasa && source venv/bin/activate && \
python scripts/train/train_model.py --train-end-date "2024-12-31" --step-count 1 && \
echo "学習完了！10秒後にインスタンスを停止します..." && \
sleep 10 && \
sudo shutdown -h now

# バックグラウンドで実行し、完了後自動停止
nohup bash -c 'cd /home/ubuntu/yamasa && source venv/bin/activate && \
python scripts/train/train_model.py --train-end-date "2024-12-31" --step-count 6 && \
echo "学習完了！インスタンスを停止します..." && \
sudo shutdown -h now' > training.log 2>&1 &
```

**注意事項**：
- インスタンスの再起動はAWSコンソールまたはAWS CLIから行う必要があります
- 長時間の学習の場合は、結果を確認してから手動で停止することを推奨
- S3への保存は自動的に行われるため、インスタンス停止後もデータは保持されます

## トラブルシューティング

### メモリ不足エラー
```bash
# バッチ処理モードで実行
python3 scripts/features/generate_features_batch_s3.py
```

### S3アクセスエラー
```bash
# AWS認証情報の確認
aws sts get-caller-identity
```

### 特徴量が見つからない
```bash
# 特徴量生成を実行
python3 scripts/features/generate_features_yamasa.py
```

## 更新履歴
- 2024/10/20: usage_type別の特徴量生成・学習を統合版スクリプトに集約
- 2024/10/20: material_summaryファイルに学習期間・予測期間の実績発生数カラムを追加
- 2024/10/20: 誤差率平均の定義を修正（material_key毎の平均値として正しく計算）
- 2024/10/15: Material Keyフィルタリング機能追加（データ量90%削減、処理時間1/10）
- 2024/10/15: 外れ値処理をオプション化（デフォルト無効で高速化）
- 2024/10/15: EC2インスタンス自動停止の手順追加
- 2024/10/09: Walk-forward validation実装、月単位予測対応
- 2024/10/09: バッチ処理によるメモリ効率化
- 2024/10/09: train_end_dateによるデータリーク防止機能追加
- 2024/10/09: Optuna統合、評価指標の詳細化