============================================================
  競馬予想AI - ローカル学習ガイド
============================================================

■ 動作環境
  - Windows 10/11
  - Python 3.10以上
  - NVIDIA GPU (CUDA対応)
  - RAM 16GB以上推奨

■ セットアップ (初回のみ)

  1. keiba-training.zip を解凍

  2. local_setup.bat をダブルクリック
     → 仮想環境作成、PyTorch(CUDA)、全依存パッケージを自動インストール

  3. 完了メッセージが表示されればOK


■ 学習実行

  コマンドプロンプトを開き、解凍したフォルダに移動:

    cd C:\path\to\keiba-training
    venv\Scripts\activate

  --- パターン1: LightGBMのみ (高速、約3分) ---

    python train.py

  --- パターン2: 3モデルブレンド (推奨、約10分) ---

    python train.py --blend

  --- パターン3: 全モデルブレンド (約20分) ---

    python train.py --blend-all

  --- パターン4: カスタム ---

    python train.py --models lgbm xgb catboost --n-splits 8 --min-year 2020


■ オプション一覧

  --blend          LightGBM + XGBoost + CatBoost の3モデル
  --blend-all      上記 + TabNet の4モデル
  --models X Y Z   使用モデルを個別指定 (lgbm / xgb / catboost / tabnet)
  --n-splits N     交差検証のfold数 (デフォルト: 5)
  --min-year YYYY  学習に使う最古年 (デフォルト: 2016)
  --no-advanced    拡張特徴量OFF (メモリ節約)
  --tune           Optunaハイパーパラメータチューニング実行
  --output NAME    出力モデル名 (デフォルト: keiba_model)


■ 出力ファイル

  models/keiba_model.pkl      学習済みモデル
  models/keiba_model_prev.pkl 前回モデルのバックアップ


■ 学習後の精度確認

  python test_accuracy.py

  python honest_backtest.py
  python honest_backtest.py --multiyear


■ サーバーへの反映

  学習済みの models/keiba_model.pkl をサーバーの models/ にコピー


■ トラブルシューティング

  Q: メモリ不足 (MemoryError)
  A: --min-year 2020 で直近データのみ使用
     --no-advanced で拡張特徴量OFF

  Q: CUDA out of memory
  A: TabNetのみGPU使用。--blend (TabNetなし) で回避

  Q: CatBoost/XGBoostがインストールできない
  A: --models lgbm のみで実行可能

============================================================
