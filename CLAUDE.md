# Claude Code 注意事項 - yamasa2プロジェクト

## ⚠️ 重要：このプロジェクトの特徴
- **material_key = store_code** （商品コードを含まない店舗単位の予測）
- **S3バケット: fiby-yamasa-prediction-2** （必須）
- **GitHubリポジトリ: pj_yamasa2**

## 🔴 最重要：S3バケットの使用について
**絶対にfiby-yamasa-predictionバケットを使用しないこと**
- 読み込み: **fiby-yamasa-prediction-2**のみ
- 書き込み: **fiby-yamasa-prediction-2**のみ

## デフォルト実行設定

| パラメータ | **必須デフォルト値** | 説明 |
|----------|------------------|------|
| **train_end_date** | **2024-12-31** | 学習データの終了日 |
| **step_count** | **1** | 予測月数（1ヶ月） |
| **テスト期間** | **2025年1月** | 自動的に決定される |

## Material Key定義の違い

### yamasa (元プロジェクト)
- material_key = store_code + "_" + product_key
- 商品・店舗レベルの予測

### yamasa2 (このプロジェクト)
- material_key = store_code
- 店舗レベルの予測
- store_code関連の特徴量（store_code×曜日など）は不要

## フィルタリング仕様

### 学習データに含める店舗（store_code）
以下の**いずれか**を満たす店舗：
1. **学習期間での実績発生数が上位N個**（Nは調整可能）
2. **テスト期間における実績発生数が step_count * 4 以上**

### 予測対象の店舗
- **テスト期間における実績発生数が step_count * 4 以上**

## 実行コマンド

### データ準備
```bash
./run_prepare.sh
```

### モデル学習
```bash
cd /home/ubuntu/yamasa2 && source venv/bin/activate && python scripts/train/train_model.py
```

### 予測実行
```bash
./run_train_predict.sh predict
```