# CLAUDE.md — keiba-predictor

競馬AI予測システム。FastAPI + scikit/LightGBM/CatBoost/XGBoost アンサンブル + 単一ページHTML UI。
PAPER TRADING のみ。ライブベット非対応。

## 起動 / 再起動

```bash
# 既存プロセス停止 + 起動
pkill -9 -f "uvicorn api_server:app" 2>/dev/null
sleep 2
nohup venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &

# ready 待ち (モデルロードが数十秒かかる)
until curl -sf http://localhost:8000/api/health 2>/dev/null | grep -q '"status":"ok"'; do
  sleep 3
  if ! ps auxww | grep -q 'uvicorn api_server:app' | grep -v "bash -c"; then
    echo DIED; tail -30 /tmp/api.log; break
  fi
done
```

## レイアウト要所

### エントリポイント
- `api_server.py` (6300行) — FastAPI アプリ本体。ルート/認証/予測/管理API/フロント配信をすべて含む
- `static/index.html` (2500行) — SPA フロントエンド (インラインJS/CSS、PWA対応)
- `main.py` — CLI エントリ (バックテスト等に使う)

### モデル / 特徴量
- `model_trainer.py` (2800行) — `RacePredictionModel`、`_BlendedModel`、予測パイプライン
- `data_processor.py` (1900行) — `RaceDataProcessor` (前処理) と `FeatureEngineer` (特徴量)
- `features_advanced.py` — GOD 層の追加特徴量 (`AdvancedFeatureEngineer.create_all`)
- `god_ensemble.py` — メタスタッキング + BMA
- `integrate_extra_data.py` — 血統/調教/予想印のマージ。`init_services` から `integrate_training` を呼ぶ
- `bloodline_embeddings.py` — 血統の学習埋め込み
- モデルファイル: `models/keiba_model.pkl` (バージョン v7.2-advanced, 220特徴量)

### データ
- `data/races_YYYY.csv` (年別) → 起動時に `_all_races.csv` に結合
- `data/bloodline.csv` (horse_id→sire/dam/sire_sire/dam_sire) → 起動時にDBへマージ
- `data/training.csv` (調教評価) → `integrate_training` でマージ
- `data/users.json` — ユーザーDB (role列あり: `"admin"` / `"user"`)
- `data/.jwt_secret` (0600) — JWT署名鍵、自動生成・永続化
- `data/.admin_initial_password` (0600) — 初回adminパスワード (初回起動後に削除推奨)

### 運用
- `auto_bettor.py` — Paper trading (`AutoBettor`)。Kelly sizing + `uncertainty_discount`
- `risk_manager.py` — サーキットブレーカー (halt, cooldown, DD, daily/weekly/consecutive limits, per-race cap)
- `honest_backtest.py` — バックテスト + メトリクス (Sharpe, bootstrap CI, profit factor, 連敗)
- `daily_scraper.py` / `scraper_v2.py` / `shutuba_scraper.py` — データ収集
- `precache_predictions.py` — 日次レースの事前予測キャッシュ

### 認証
- JWT (HS256, 24h有効) — `get_current_user` で依存注入
- `_require_admin(current_user)` — `role == "admin"` で判定
- ログイン: username単位 5回/5分 + IP単位 20回/5分 のレート制限

## 予測パイプライン (重要)

**GET `/api/predict/{race_id}` (と POST `/api/predict`) は同じパイプライン**:

```
1. race_df = db.df[race_id]                        # 対象レース
2. context_df = db.df[horse_id in race_horses]     # 各馬の過去全データ
3. combined = concat([context_df, race_df])        # 履歴込み
4. リークマスク: finish_position/time/prize_money/passing_order/last_3f を対象レース側でNull化
5. RaceDataProcessor.preprocess → FeatureEngineer.create_features → AdvancedFeatureEngineer.create_all
6. featured[race_id == target]                     # 対象レースのみ抽出
7. model.predict(featured)                          # EV分布 + Conformal CI 付き
```

**過去: GET版がこの履歴マージを怠って 25+ 特徴量を 0埋めしていた → 修正済 (本セッション)。**

## 予測レスポンスの不確実性フィールド

- `ev_mean`, `ev_std`, `ev_p25`, `ev_p75` — モンテカルロ EV分布
- `pred_rank_lower80 / upper80` — 予測着順80%CI (UI主表示)
- `pred_rank_lower95 / upper95` — 95%CI (外側薄色)
- CI はレース頭数でクリップ済み

## テスト

```bash
venv/bin/python3 -m pytest tests/ -v
```
- `test_risk_manager.py` — サーキットブレーカー (8)
- `test_bloodline_features.py` — 血統特徴量 (4)
- `test_kelly_sizing.py` — Kelly + uncertainty discount (7)
- `test_backtest_metrics.py` — バックテスト指標 (7)
- `test_api.py` — 生きているサーバ前提 (一部失敗はサーバ非起動時に起こる)

## よくある落とし穴

- **`model.load("keiba_model")`** は拡張子なし・`model_dir` 相対 (`models/keiba_model.pkl` を探す)
- **`pkill`+`nohup`** を同じコマンドチェーンで繋ぐと exit code 144 で死ぬことがある。分離推奨
- **`_all_races.csv` は起動時に必ず再生成される** (races_YYYY.csv + bloodline + training からビルド)
- **Conformal Q95 ≈ 6.7** → 13頭レースで95%CIはほぼ全範囲。80%CIが実用的
- **Stackingフォールバック判定**: 絶対閾値 (<0.05 が過半) ではなく std<1e-3 で判定する (1/16≈6%が正例率なので)
- **StaticFiles** は `/static/*` にマウントされるが、PWA資産 (`sw.js`, `manifest.json`, `icon.svg`) は逆プロキシ経由でルートに来るため個別ルートで配信

## ログ確認スニペット

```bash
tail -50 /tmp/api.log
grep -iE "ERROR|WARNING|特徴量.*不足" /tmp/api.log | tail -20
grep -iE "Bloodline merged|Training data merged" /tmp/api.log
```

## 管理画面 (admin only)

- `/#admin` にアクセス (ナビに "🛡️ 管理" が表示される、`S.isAdmin` が true のときのみ)
- セクション: リスク管理(halt/resume) / システム / モデル / スクレイパー / キャッシュ
- API: `POST /api/admin/risk/halt` {reason}, `POST /api/admin/risk/resume`

## 環境変数

- `KEIBA_JWT_SECRET` — **本番必須** (未設定時は `data/.jwt_secret` に自動生成)
- `KEIBA_ADMIN_PASSWORD` — 初回admin作成時のみ使用
- `KEIBA_DATA_DIR` — デフォルト `data`
- `PORT` — デフォルト 8000
