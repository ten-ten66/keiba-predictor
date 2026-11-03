"""
競馬予想AI - 認証付きAPIルーター
=================================
api_server.py に組み込む認証エンドポイント。

使い方:
  from auth_routes import auth_router, get_current_user
  app.include_router(auth_router)

  @app.get("/api/my-bets")
  async def my_bets(user = Depends(get_current_user)):
      tracker = get_user_tracker(user.user_id)
      ...
"""

import os
import sys
from typing import Optional
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import APIRouter, HTTPException, Depends, Header
    from pydantic import BaseModel
except ImportError:
    print("❌ pip install fastapi")
    sys.exit(1)

from auth import UserDB, UserNotifier, create_token, decode_token, NotificationSettings
from betting_tracker import BettingTracker

# ============================================================
# 初期化
# ============================================================

user_db = UserDB()
user_notifier = UserNotifier(user_db)
auth_router = APIRouter(prefix="/auth", tags=["認証"])

# ユーザー別トラッカーのキャッシュ
_user_trackers = {}


# ============================================================
# リクエストモデル
# ============================================================

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdateRequest(BaseModel):
    display_name: str = None
    bankroll: int = None
    kelly_fraction: float = None
    default_bet_amount: int = None

class NotificationUpdateRequest(BaseModel):
    discord_webhook: str = None
    line_token: str = None
    telegram_token: str = None
    telegram_chat_id: str = None
    notify_daily_prediction: bool = None
    notify_odds_alert: bool = None
    notify_race_result: bool = None
    notify_pre_race: bool = None
    notify_daily_summary: bool = None

class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str

class UserBetRequest(BaseModel):
    race_id: str
    race_name: str
    race_date: str
    venue: str
    bet_type: str
    selections: list
    amount: int = 100
    is_ai_recommended: bool = False
    memo: str = ""


# ============================================================
# 認証ヘルパー
# ============================================================

async def get_current_user(authorization: Optional[str] = Header(None)):
    """認証ミドルウェア: Authorizationヘッダーからユーザーを取得"""
    if not authorization:
        raise HTTPException(401, "認証が必要です。ログインしてください。")

    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    payload = decode_token(token)

    if not payload:
        raise HTTPException(401, "トークンが無効または期限切れです。再ログインしてください。")

    user = user_db.get_user(payload.get("sub", ""))
    if not user or not user.is_active:
        raise HTTPException(401, "ユーザーが見つかりません")

    return user


async def get_optional_user(authorization: Optional[str] = Header(None)):
    """認証オプショナル: ログインしてなくても通す"""
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


def get_user_tracker(user_id: str) -> BettingTracker:
    """ユーザー別のBettingTrackerを取得"""
    if user_id not in _user_trackers:
        user_dir = user_db.get_user_data_dir(user_id)
        _user_trackers[user_id] = BettingTracker(data_dir=user_dir)
    return _user_trackers[user_id]


# ============================================================
# 認証エンドポイント
# ============================================================

@auth_router.post("/register")
async def register(req: RegisterRequest):
    """アカウント作成"""
    try:
        user = user_db.register(req.email, req.password, req.display_name)
        access = create_token(user.user_id, "access")
        refresh = create_token(user.user_id, "refresh")
        return {
            "status": "ok",
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
            },
            "access_token": access,
            "refresh_token": refresh,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@auth_router.post("/login")
async def login(req: LoginRequest):
    """ログイン"""
    user = user_db.login(req.email, req.password)
    if not user:
        raise HTTPException(401, "メールアドレスまたはパスワードが正しくありません")

    access = create_token(user.user_id, "access")
    refresh = create_token(user.user_id, "refresh")
    return {
        "status": "ok",
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.display_name,
        },
        "access_token": access,
        "refresh_token": refresh,
    }


@auth_router.post("/refresh")
async def refresh_token(authorization: Optional[str] = Header(None)):
    """トークン更新"""
    if not authorization:
        raise HTTPException(401, "リフレッシュトークンが必要です")

    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(401, "無効なリフレッシュトークンです")

    user = user_db.get_user(payload.get("sub", ""))
    if not user:
        raise HTTPException(401, "ユーザーが見つかりません")

    new_access = create_token(user.user_id, "access")
    return {"access_token": new_access}


@auth_router.get("/me")
async def get_me(user=Depends(get_current_user)):
    """自分のプロフィール取得"""
    return {
        "user_id": user.user_id,
        "email": user.email,
        "display_name": user.display_name,
        "bankroll": user.bankroll,
        "kelly_fraction": user.kelly_fraction,
        "default_bet_amount": user.default_bet_amount,
        "created_at": user.created_at,
        "last_login": user.last_login,
        "total_bets": user.total_bets,
        "total_profit": user.total_profit,
        "notifications": asdict(user.notifications),
    }


@auth_router.put("/me")
async def update_me(req: ProfileUpdateRequest, user=Depends(get_current_user)):
    """プロフィール更新"""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    updated = user_db.update_profile(user.user_id, updates)
    if not updated:
        raise HTTPException(500, "更新に失敗しました")
    return {"status": "ok", "display_name": updated.display_name, "bankroll": updated.bankroll}


@auth_router.put("/me/notifications")
async def update_notifications(req: NotificationUpdateRequest, user=Depends(get_current_user)):
    """通知設定更新"""
    settings = {k: v for k, v in req.dict().items() if v is not None}
    updated = user_db.update_notifications(user.user_id, settings)
    if not updated:
        raise HTTPException(500, "更新に失敗しました")
    return {"status": "ok", "notifications": asdict(updated)}


@auth_router.put("/me/password")
async def change_password(req: PasswordChangeRequest, user=Depends(get_current_user)):
    """パスワード変更"""
    ok = user_db.update_password(user.user_id, req.old_password, req.new_password)
    if not ok:
        raise HTTPException(400, "現在のパスワードが正しくないか、新しいパスワードが短すぎます")
    return {"status": "ok"}


@auth_router.delete("/me")
async def delete_me(user=Depends(get_current_user)):
    """アカウント削除"""
    # キャッシュからも削除
    _user_trackers.pop(user.user_id, None)
    ok = user_db.delete_user(user.user_id)
    if not ok:
        raise HTTPException(500, "削除に失敗しました")
    return {"status": "ok", "message": "アカウントを削除しました"}


# ============================================================
# ユーザー別データエンドポイント
# ============================================================

@auth_router.get("/me/balance")
async def my_balance(period: str = "all", user=Depends(get_current_user)):
    """自分の収支"""
    tracker = get_user_tracker(user.user_id)
    race_date = None
    if period == "today":
        from datetime import date
        race_date = date.today().isoformat()
    return tracker.get_summary(race_date=race_date)


@auth_router.get("/me/history")
async def my_history(limit: int = 50, offset: int = 0, user=Depends(get_current_user)):
    """自分の購入履歴"""
    tracker = get_user_tracker(user.user_id)
    bets = tracker.bets[offset:offset + limit]
    return {
        "total": len(tracker.bets),
        "bets": [asdict(b) for b in reversed(bets)],
    }


@auth_router.post("/me/bets")
async def add_my_bet(req: UserBetRequest, user=Depends(get_current_user)):
    """自分の馬券を登録"""
    tracker = get_user_tracker(user.user_id)
    try:
        bet = tracker.add_bet(
            race_id=req.race_id, race_name=req.race_name,
            race_date=req.race_date, venue=req.venue,
            bet_type=req.bet_type, selections=req.selections,
            amount=req.amount, is_ai_recommended=req.is_ai_recommended,
            memo=req.memo,
        )
        # ユーザーの累計更新
        user_db.users[user.user_id].total_bets += 1
        user_db._save()
        return {"status": "ok", "bet": asdict(bet)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@auth_router.get("/me/race/{race_id}")
async def my_race_detail(race_id: str, user=Depends(get_current_user)):
    """自分の特定レースの詳細"""
    tracker = get_user_tracker(user.user_id)
    return tracker.get_race_detail(race_id)


@auth_router.get("/me/export/csv")
async def export_csv(user=Depends(get_current_user)):
    """収支CSVエクスポート"""
    tracker = get_user_tracker(user.user_id)
    user_dir = user_db.get_user_data_dir(user.user_id)
    path = tracker.export_history_csv(os.path.join(user_dir, "history.csv"))
    return {"status": "ok", "path": path}


# ============================================================
# 通知テスト
# ============================================================

@auth_router.post("/me/notifications/test")
async def test_notification(user=Depends(get_current_user)):
    """通知テスト送信"""
    results = user_notifier.send_to_user(
        user.user_id,
        f"🏇 KEIBA AI テスト通知\n{user.display_name}さんの通知設定が正常に動作しています。",
        notification_type="general",
    )
    return {"status": "ok", "results": results}
