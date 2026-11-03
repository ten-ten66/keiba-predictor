"""
競馬予想AI - ユーザー認証 & マルチユーザー管理
================================================
アカウント作成・ログイン・ユーザー別データ分離。

機能:
  1. アカウント登録（メール+パスワード）
  2. JWT認証（アクセストークン + リフレッシュトークン）
  3. ユーザー別データディレクトリ
  4. ユーザー別の通知設定（Discord/LINE/Telegram）
  5. パスワードハッシュ化（bcrypt）

APIエンドポイント:
  POST /auth/register         アカウント作成
  POST /auth/login            ログイン → JWT取得
  POST /auth/refresh          トークン更新
  GET  /auth/me               ユーザー情報取得
  PUT  /auth/me               プロフィール更新
  PUT  /auth/me/notifications 通知設定更新
  DELETE /auth/me             アカウント削除
"""

import os
import json
import hashlib
import hmac
import secrets
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

# JWT
try:
    import jwt as pyjwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    # 簡易JWT（pyjwtなしでも動くフォールバック）
    import base64

# パスワードハッシュ
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


# ============================================================
# 設定
# ============================================================

AUTH_CONFIG = {
    "secret_key": os.environ.get("KEIBA_SECRET_KEY", secrets.token_hex(32)),
    "algorithm": "HS256",
    "access_token_expire_minutes": 60 * 24,     # 24時間
    "refresh_token_expire_days": 30,
    "users_dir": os.environ.get("KEIBA_DATA_DIR", "data") + "/users",
    "users_db_file": os.environ.get("KEIBA_DATA_DIR", "data") + "/users.json",
}


# ============================================================
# データモデル
# ============================================================

@dataclass
class NotificationSettings:
    """ユーザー別の通知設定"""
    discord_webhook: str = ""
    line_token: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""
    # 通知タイミング
    notify_daily_prediction: bool = True     # 毎朝の予想
    notify_odds_alert: bool = True           # オッズ急変動
    notify_race_result: bool = True          # レース結果
    notify_pre_race: bool = True             # レース前アラート
    notify_daily_summary: bool = True        # 日次まとめ


@dataclass
class UserProfile:
    """ユーザープロフィール"""
    user_id: str
    email: str
    display_name: str
    password_hash: str
    created_at: str
    last_login: str = ""
    is_active: bool = True
    bankroll: int = 100000                   # デフォルト資金
    kelly_fraction: float = 0.25             # ケリー基準の割合
    default_bet_amount: int = 100
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    # メタ
    total_bets: int = 0
    total_profit: float = 0


# ============================================================
# パスワードハッシュ
# ============================================================

def hash_password(password: str) -> str:
    """パスワードをハッシュ化"""
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    else:
        # bcryptなしのフォールバック（SHA-256 + salt）
        salt = secrets.token_hex(16)
        h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return f"sha256:{salt}:{h}"


def verify_password(password: str, hashed: str) -> bool:
    """パスワードを検証"""
    if HAS_BCRYPT and not hashed.startswith("sha256:"):
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    elif hashed.startswith("sha256:"):
        _, salt, h = hashed.split(":")
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == h
    return False


# ============================================================
# JWT トークン
# ============================================================

def create_token(user_id: str, token_type: str = "access") -> str:
    """JWTトークンを生成"""
    if token_type == "access":
        exp = datetime.utcnow() + timedelta(minutes=AUTH_CONFIG["access_token_expire_minutes"])
    else:
        exp = datetime.utcnow() + timedelta(days=AUTH_CONFIG["refresh_token_expire_days"])

    payload = {
        "sub": user_id,
        "type": token_type,
        "exp": exp.timestamp(),
        "iat": datetime.utcnow().timestamp(),
    }

    if HAS_JWT:
        return pyjwt.encode(payload, AUTH_CONFIG["secret_key"], algorithm=AUTH_CONFIG["algorithm"])
    else:
        # 簡易JWT（pyjwtなしフォールバック）
        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig_input = f"{header}.{payload_b64}"
        sig = hmac.new(AUTH_CONFIG["secret_key"].encode(), sig_input.encode(), hashlib.sha256).hexdigest()
        return f"{header}.{payload_b64}.{sig}"


def decode_token(token: str) -> Optional[Dict]:
    """JWTトークンをデコード"""
    try:
        if HAS_JWT:
            payload = pyjwt.decode(token, AUTH_CONFIG["secret_key"], algorithms=[AUTH_CONFIG["algorithm"]])
        else:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            # 署名検証
            sig_input = f"{parts[0]}.{parts[1]}"
            expected_sig = hmac.new(AUTH_CONFIG["secret_key"].encode(), sig_input.encode(), hashlib.sha256).hexdigest()
            if parts[2] != expected_sig:
                return None
            # ペイロードデコード
            padding = 4 - len(parts[1]) % 4
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * padding))

        # 有効期限チェック
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            return None

        return payload
    except Exception:
        return None


# ============================================================
# ユーザーDB（JSONファイルベース）
# ============================================================

class UserDB:
    """ユーザー管理データベース"""

    def __init__(self):
        self.db_file = AUTH_CONFIG["users_db_file"]
        self.users_dir = AUTH_CONFIG["users_dir"]
        self.users: Dict[str, UserProfile] = {}
        self._email_index: Dict[str, str] = {}  # email -> user_id
        self._load()

    def _load(self):
        """DBファイルを読み込み"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for uid, udata in data.items():
                    notif = udata.pop("notifications", {})
                    user = UserProfile(**udata, notifications=NotificationSettings(**notif))
                    self.users[uid] = user
                    self._email_index[user.email.lower()] = uid
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"⚠️ ユーザーDB読み込みエラー: {e}")

    def _save(self):
        """DBファイルを保存"""
        os.makedirs(os.path.dirname(self.db_file) or ".", exist_ok=True)
        data = {}
        for uid, user in self.users.items():
            ud = asdict(user)
            data[uid] = ud

        # アトミック保存
        tmp = self.db_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.db_file)

    def register(self, email: str, password: str, display_name: str = "") -> UserProfile:
        """アカウント作成"""
        email_lower = email.lower().strip()

        # バリデーション
        if not email_lower or "@" not in email_lower:
            raise ValueError("有効なメールアドレスを入力してください")
        if email_lower in self._email_index:
            raise ValueError("このメールアドレスは既に登録されています")
        if len(password) < 6:
            raise ValueError("パスワードは6文字以上必要です")

        # ユーザーID生成
        user_id = f"U{secrets.token_hex(8)}"

        # ユーザー作成
        user = UserProfile(
            user_id=user_id,
            email=email_lower,
            display_name=display_name or email_lower.split("@")[0],
            password_hash=hash_password(password),
            created_at=datetime.now().isoformat(),
        )

        self.users[user_id] = user
        self._email_index[email_lower] = user_id

        # ユーザー別ディレクトリ作成
        user_dir = os.path.join(self.users_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)

        self._save()
        return user

    def login(self, email: str, password: str) -> Optional[UserProfile]:
        """ログイン（認証）"""
        email_lower = email.lower().strip()
        user_id = self._email_index.get(email_lower)
        if not user_id:
            return None

        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None

        if not verify_password(password, user.password_hash):
            return None

        # 最終ログイン更新
        user.last_login = datetime.now().isoformat()
        self._save()
        return user

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """ユーザー取得"""
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """メールでユーザー取得"""
        uid = self._email_index.get(email.lower().strip())
        return self.users.get(uid) if uid else None

    def update_profile(self, user_id: str, updates: Dict) -> Optional[UserProfile]:
        """プロフィール更新"""
        user = self.users.get(user_id)
        if not user:
            return None

        allowed = {"display_name", "bankroll", "kelly_fraction", "default_bet_amount"}
        for key, val in updates.items():
            if key in allowed:
                setattr(user, key, val)

        self._save()
        return user

    def update_notifications(self, user_id: str, settings: Dict) -> Optional[NotificationSettings]:
        """通知設定更新"""
        user = self.users.get(user_id)
        if not user:
            return None

        for key, val in settings.items():
            if hasattr(user.notifications, key):
                setattr(user.notifications, key, val)

        self._save()
        return user.notifications

    def update_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """パスワード変更"""
        user = self.users.get(user_id)
        if not user:
            return False
        if not verify_password(old_password, user.password_hash):
            return False
        if len(new_password) < 6:
            return False

        user.password_hash = hash_password(new_password)
        self._save()
        return True

    def delete_user(self, user_id: str) -> bool:
        """アカウント削除"""
        user = self.users.get(user_id)
        if not user:
            return False

        # データディレクトリ削除
        user_dir = os.path.join(self.users_dir, user_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)

        # インデックスから削除
        self._email_index.pop(user.email.lower(), None)
        del self.users[user_id]
        self._save()
        return True

    def get_user_data_dir(self, user_id: str) -> str:
        """ユーザー別データディレクトリのパスを返す"""
        d = os.path.join(self.users_dir, user_id)
        os.makedirs(d, exist_ok=True)
        return d

    def list_users(self) -> List[Dict]:
        """全ユーザー一覧（管理用）"""
        return [
            {
                "user_id": u.user_id,
                "email": u.email,
                "display_name": u.display_name,
                "is_active": u.is_active,
                "created_at": u.created_at,
                "last_login": u.last_login,
                "total_bets": u.total_bets,
            }
            for u in self.users.values()
        ]


# ============================================================
# ユーザー別通知
# ============================================================

class UserNotifier:
    """ユーザー別の通知送信"""

    def __init__(self, user_db: UserDB):
        self.db = user_db

    def send_to_user(self, user_id: str, message: str, notification_type: str = "general") -> Dict[str, bool]:
        """特定ユーザーに通知"""
        user = self.db.get_user(user_id)
        if not user:
            return {}

        ns = user.notifications
        results = {}

        # 通知タイプに応じてフィルタ
        type_flags = {
            "daily_prediction": ns.notify_daily_prediction,
            "odds_alert": ns.notify_odds_alert,
            "race_result": ns.notify_race_result,
            "pre_race": ns.notify_pre_race,
            "daily_summary": ns.notify_daily_summary,
            "general": True,
        }
        if not type_flags.get(notification_type, True):
            return {"skipped": True}

        # Discord
        if ns.discord_webhook:
            results["discord"] = self._send_discord(ns.discord_webhook, message)

        # LINE
        if ns.line_token:
            results["line"] = self._send_line(ns.line_token, message)

        # Telegram
        if ns.telegram_token and ns.telegram_chat_id:
            results["telegram"] = self._send_telegram(ns.telegram_token, ns.telegram_chat_id, message)

        return results

    def broadcast_all(self, message: str, notification_type: str = "general") -> Dict[str, Dict]:
        """全ユーザーに通知"""
        results = {}
        for uid in self.db.users:
            results[uid] = self.send_to_user(uid, message, notification_type)
        return results

    @staticmethod
    def _send_discord(webhook_url: str, message: str) -> bool:
        import requests
        try:
            r = requests.post(webhook_url, json={"content": message, "username": "🏇 KEIBA AI"}, timeout=10)
            return r.status_code == 204
        except Exception:
            return False

    @staticmethod
    def _send_line(token: str, message: str) -> bool:
        import requests
        try:
            r = requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {token}"},
                data={"message": message}, timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _send_telegram(token: str, chat_id: str, message: str) -> bool:
        import requests
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False


# ============================================================
# FastAPI 認証ミドルウェア（api_server.py で使用）
# ============================================================

def get_current_user_id(authorization: str = None) -> Optional[str]:
    """
    Authorizationヘッダーからユーザーを取得。
    FastAPIの Depends() で使用。

    Usage:
        from auth import get_current_user_id
        @app.get("/api/my-data")
        async def my_data(user_id: str = Depends(get_current_user_id)):
            ...
    """
    if not authorization:
        return None

    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    payload = decode_token(token)
    if not payload:
        return None

    return payload.get("sub")


# ============================================================
# テスト
# ============================================================

if __name__ == "__main__":
    import shutil

    print("=" * 50)
    print("🔐 ユーザー認証テスト")
    print("=" * 50)

    # テスト用ディレクトリ
    test_dir = "data/test_auth"
    AUTH_CONFIG["users_dir"] = test_dir + "/users"
    AUTH_CONFIG["users_db_file"] = test_dir + "/users.json"
    os.makedirs(test_dir, exist_ok=True)

    db = UserDB()

    # 1. 登録
    print("\n[1] アカウント作成")
    user1 = db.register("test@example.com", "password123", "テストユーザー")
    print(f"  ✅ 登録: {user1.user_id} / {user1.email} / {user1.display_name}")

    user2 = db.register("user2@example.com", "mypassword", "ユーザー2")
    print(f"  ✅ 登録: {user2.user_id} / {user2.email}")

    # 重複チェック
    try:
        db.register("test@example.com", "other", "重複")
        print("  ❌ 重複登録が通った")
    except ValueError as e:
        print(f"  ✅ 重複拒否: {e}")

    # パスワード短すぎ
    try:
        db.register("short@example.com", "12345", "短い")
        print("  ❌ 短いパスワードが通った")
    except ValueError:
        print(f"  ✅ 短いパスワード拒否")

    # 2. ログイン
    print("\n[2] ログイン")
    logged_in = db.login("test@example.com", "password123")
    print(f"  ✅ ログイン成功: {logged_in.user_id}")

    failed = db.login("test@example.com", "wrongpassword")
    print(f"  ✅ 不正パスワード拒否: {failed is None}")

    failed2 = db.login("noexist@example.com", "password")
    print(f"  ✅ 存在しないユーザー拒否: {failed2 is None}")

    # 3. JWT
    print("\n[3] JWTトークン")
    access = create_token(user1.user_id, "access")
    print(f"  トークン: {access[:50]}...")

    payload = decode_token(access)
    print(f"  ✅ デコード: user_id={payload['sub']}")

    # 改ざんチェック
    tampered = access[:-5] + "XXXXX"
    result = decode_token(tampered)
    print(f"  ✅ 改ざん検出: {result is None}")

    # 4. プロフィール更新
    print("\n[4] プロフィール更新")
    db.update_profile(user1.user_id, {"display_name": "新しい名前", "bankroll": 50000})
    updated = db.get_user(user1.user_id)
    print(f"  ✅ 名前: {updated.display_name} / 資金: ¥{updated.bankroll:,}")

    # 5. 通知設定
    print("\n[5] 通知設定")
    db.update_notifications(user1.user_id, {
        "discord_webhook": "https://example.invalid/webhook",
        "notify_odds_alert": False,
    })
    ns = db.get_user(user1.user_id).notifications
    print(f"  ✅ Discord: {ns.discord_webhook[:30]}...")
    print(f"  ✅ オッズアラート: {ns.notify_odds_alert}")

    # 6. ユーザー別ディレクトリ
    print("\n[6] ユーザー別データ")
    dir1 = db.get_user_data_dir(user1.user_id)
    dir2 = db.get_user_data_dir(user2.user_id)
    print(f"  ✅ User1: {dir1}")
    print(f"  ✅ User2: {dir2}")
    assert dir1 != dir2

    # 7. パスワード変更
    print("\n[7] パスワード変更")
    ok = db.update_password(user1.user_id, "password123", "newpassword456")
    print(f"  ✅ 変更成功: {ok}")
    ok = db.login("test@example.com", "newpassword456")
    print(f"  ✅ 新パスワードでログイン: {ok is not None}")

    # 8. ユーザー一覧
    print(f"\n[8] ユーザー一覧: {len(db.list_users())}名")
    for u in db.list_users():
        print(f"  {u['user_id']} / {u['email']} / {u['display_name']}")

    # 9. 削除
    print("\n[9] アカウント削除")
    db.delete_user(user2.user_id)
    print(f"  ✅ 削除後: {len(db.list_users())}名")

    # 10. DB永続化テスト
    print("\n[10] 永続化テスト")
    db2 = UserDB()
    restored = db2.get_user(user1.user_id)
    print(f"  ✅ 再読込: {restored.display_name} / {restored.email}")

    # クリーンアップ
    shutil.rmtree(test_dir)
    print("\n🏁 全テスト完了")
