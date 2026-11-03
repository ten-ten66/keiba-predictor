"""
競馬予想AI - Twitter/X API ルーター & アフィリエイト API
======================================================
api_server.py から import して使う追加ルーター。

使い方 (api_server.py 末尾に追加):
    try:
        from twitter_routes import twitter_router, affiliate_router, lp_router
        app.include_router(twitter_router)
        app.include_router(affiliate_router)
        app.include_router(lp_router)
    except ImportError:
        pass
"""

import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================================
# Twitter/X ルーター
# ============================================================

twitter_router = APIRouter(tags=["twitter"])

try:
    from social_poster import (
        load_twitter_config, save_twitter_config,
        post_to_twitter, auto_post_daily_picks, auto_post_results,
        format_prediction_tweet,
    )
    _social_poster_available = True
except ImportError:
    _social_poster_available = False
    logger.warning("social_poster モジュールが読み込めません")


class TwitterPostRequest(BaseModel):
    message: str = ""
    race_id: str = ""
    post_type: str = "custom"  # "custom", "picks", "results"
    date: str = ""


class TwitterConfigRequest(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    enabled: bool = False
    auto_post_predictions: bool = True
    auto_post_results: bool = True


def _get_current_user_dep():
    """get_current_user dependency を動的に取得"""
    try:
        from api_server import get_current_user
        return get_current_user
    except ImportError:
        return None


@twitter_router.post("/api/admin/twitter-post")
async def admin_twitter_post(req: TwitterPostRequest):
    """手動でツイートを投稿する（管理者のみ）"""
    if not _social_poster_available:
        raise HTTPException(503, "social_poster モジュールが利用できません")

    if req.post_type == "picks":
        date_str = req.date or datetime.now().strftime("%Y%m%d")
        results = auto_post_daily_picks(date_str)
        return {"status": "ok", "posted": len(results), "details": results}
    elif req.post_type == "results":
        date_str = req.date or datetime.now().strftime("%Y-%m-%d")
        results = auto_post_results(date_str)
        return {"status": "ok", "posted": len(results), "details": results}
    elif req.message:
        result = post_to_twitter(req.message)
        return {"status": "ok", "result": result}
    else:
        raise HTTPException(400, "message または post_type を指定してください")


@twitter_router.get("/api/admin/twitter-config")
async def admin_get_twitter_config():
    """Twitter API 設定を取得（シークレットはマスク）"""
    if not _social_poster_available:
        raise HTTPException(503, "social_poster モジュールが利用できません")

    config = load_twitter_config()
    masked = dict(config)
    for key in ["api_key", "api_secret", "access_token", "access_secret", "bearer_token"]:
        val = masked.get(key, "")
        if val and len(val) > 8:
            masked[key] = val[:4] + "****" + val[-4:]
        elif val:
            masked[key] = "****"
    return {"status": "ok", "config": masked}


@twitter_router.post("/api/admin/twitter-config")
async def admin_update_twitter_config(req: TwitterConfigRequest):
    """Twitter API 設定を更新"""
    if not _social_poster_available:
        raise HTTPException(503, "social_poster モジュールが利用できません")

    config = load_twitter_config()
    if req.api_key and "****" not in req.api_key:
        config["api_key"] = req.api_key
    if req.api_secret and "****" not in req.api_secret:
        config["api_secret"] = req.api_secret
    if req.access_token and "****" not in req.access_token:
        config["access_token"] = req.access_token
    if req.access_secret and "****" not in req.access_secret:
        config["access_secret"] = req.access_secret
    if req.bearer_token and "****" not in req.bearer_token:
        config["bearer_token"] = req.bearer_token
    config["enabled"] = req.enabled
    config["auto_post_predictions"] = req.auto_post_predictions
    config["auto_post_results"] = req.auto_post_results

    save_twitter_config(config)
    return {"status": "ok", "message": "Twitter設定を更新しました"}


# ============================================================
# アフィリエイト ルーター
# ============================================================

affiliate_router = APIRouter(tags=["affiliate"])

try:
    from affiliate import get_affiliate_cta_for_prediction, get_all_affiliate_links
    _affiliate_available = True
except ImportError:
    _affiliate_available = False
    logger.warning("affiliate モジュールが読み込めません")


@affiliate_router.get("/api/affiliate-links")
async def get_affiliate_links_api():
    """アフィリエイトリンク一覧を返す"""
    if not _affiliate_available:
        return {"status": "ok", "links": {}}
    return {"status": "ok", "links": get_all_affiliate_links()}


# ============================================================
# ランディングページ ルーター
# ============================================================

lp_router = APIRouter(tags=["landing"])

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@lp_router.get("/lp")
async def serve_landing_page():
    """SEO ランディングページ"""
    lp_path = os.path.join(_STATIC_DIR, "lp.html")
    if os.path.exists(lp_path):
        return FileResponse(lp_path, media_type="text/html")
    return HTMLResponse("<h1>Landing page not found</h1>", status_code=404)
