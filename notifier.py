"""
競馬予想AI - マルチチャネル通知
================================
Discord / LINE にユーザーごとの設定で通知を送信。

通知設定は data/notification_settings.json にユーザーごとに保存。
各ユーザーが LINE トークン / Discord Webhook URL を登録し、
希望する通知タイプ（予測/高期待値/結果）を選択できる。
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NOTIFICATION_SETTINGS_FILE = os.path.join(
    os.environ.get("KEIBA_DATA_DIR", "data"), "notification_settings.json"
)

DEFAULT_SETTINGS = {
    "line_token": "",
    "discord_webhook": "",
    "notify_predict": True,
    "notify_value_bet": True,
    "notify_result": False,
    # フィルタ設定
    "bet_types": [],           # 空=全券種, 例: ["単勝","複勝","馬連"]
    "min_ev": 1.2,             # 最低期待値（これ以上のみ通知）
    "venues": [],              # 空=全競馬場, 例: ["東京","中山"]
    "grades": [],              # 空=全グレード, 例: ["G1","G2","G3"]
    "favorite_horses": [],     # お気に入り馬名リスト
}


# ============================================================
# 設定の読み書き
# ============================================================

def _load_all_settings() -> dict:
    if not os.path.exists(NOTIFICATION_SETTINGS_FILE):
        return {}
    try:
        with open(NOTIFICATION_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_all_settings(data: dict):
    os.makedirs(os.path.dirname(NOTIFICATION_SETTINGS_FILE), exist_ok=True)
    with open(NOTIFICATION_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_user_settings(username: str) -> dict:
    """ユーザーの通知設定を取得"""
    all_s = _load_all_settings()
    user_s = all_s.get(username, {})
    result = dict(DEFAULT_SETTINGS)
    result.update(user_s)
    return result


def save_user_settings(username: str, settings: dict):
    """ユーザーの通知設定を保存"""
    all_s = _load_all_settings()
    all_s[username] = settings
    _save_all_settings(all_s)


# ============================================================
# 送信チャネル
# ============================================================

def send_line(token: str, message: str) -> bool:
    """LINE Notify で通知を送信"""
    if not token:
        return False
    try:
        import requests
        resp = requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("LINE通知送信成功")
            return True
        logger.warning(f"LINE通知エラー: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"LINE通知例外: {e}")
        return False


def send_discord(webhook_url: str, message: str) -> bool:
    """Discord Webhook で通知を送信"""
    if not webhook_url:
        return False
    try:
        import requests
        resp = requests.post(
            webhook_url,
            json={"content": message, "username": "KEIBA AI"},
            timeout=10,
        )
        if resp.status_code in (200, 204):
            logger.info("Discord通知送信成功")
            return True
        logger.warning(f"Discord通知エラー: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Discord通知例外: {e}")
        return False


# ============================================================
# ユーザー通知
# ============================================================

def should_notify(settings: dict, event_type: str, context: Optional[Dict] = None) -> bool:
    """
    フィルタ設定に基づいて通知すべきか判定。

    context (optional):
        bet_type: str       - 券種名 ("単勝", "馬連", etc.)
        ev: float           - 期待値
        venue: str          - 競馬場名
        grade: str          - グレード ("G1", "G2", "G3", "OP", "条件")
        horse_names: list   - 出走馬名リスト
    """
    # 通知タイプの有効/無効チェック
    event_key = f"notify_{event_type}"
    if not settings.get(event_key, False):
        return False

    if not context:
        return True

    # 券種フィルタ
    allowed_types = settings.get("bet_types", [])
    if allowed_types and context.get("bet_type"):
        if context["bet_type"] not in allowed_types:
            return False

    # 期待値しきい値
    min_ev = settings.get("min_ev", 0)
    if min_ev and context.get("ev") is not None:
        if context["ev"] < min_ev:
            return False

    # 競馬場フィルタ
    allowed_venues = settings.get("venues", [])
    if allowed_venues and context.get("venue"):
        if context["venue"] not in allowed_venues:
            return False

    # グレードフィルタ
    allowed_grades = settings.get("grades", [])
    if allowed_grades and context.get("grade"):
        if context["grade"] not in allowed_grades:
            return False

    return True


def check_favorite_horses(settings: dict, horse_names: List[str]) -> List[str]:
    """お気に入り馬が出走リストに含まれるかチェック"""
    favorites = settings.get("favorite_horses", [])
    if not favorites:
        return []
    return [h for h in horse_names if any(fav in h for fav in favorites)]


def notify_user(username: str, event_type: str, message: str,
                context: Optional[Dict] = None) -> Dict[str, bool]:
    """
    ユーザーの設定に応じて通知を送信。

    event_type: "predict" | "value_bet" | "result"
    context: フィルタ判定用の追加情報（bet_type, ev, venue, grade, horse_names）
    """
    settings = get_user_settings(username)
    results = {"line": False, "discord": False}

    if not should_notify(settings, event_type, context):
        logger.debug(f"通知スキップ: {username} {event_type} (フィルタ)")
        return results

    line_token = settings.get("line_token", "").strip()
    discord_url = settings.get("discord_webhook", "").strip()

    if line_token:
        results["line"] = send_line(line_token, message)

    if discord_url:
        results["discord"] = send_discord(discord_url, message)

    return results


def notify_all_users(event_type: str, message: str) -> Dict[str, Dict[str, bool]]:
    """全ユーザーに通知（設定に応じてフィルタ）"""
    all_s = _load_all_settings()
    results = {}
    for username in all_s:
        results[username] = notify_user(username, event_type, message)
    return results


# ============================================================
# メッセージフォーマッター
# ============================================================

def format_prediction_message(race_name: str, predictions: list) -> str:
    """予測結果の通知メッセージ"""
    lines = [f"\n🏇 AI予測完了: {race_name}"]
    for i, p in enumerate(predictions[:5], 1):
        horse = p.get("horse_name", f"#{p.get('horse_number', '?')}")
        score = p.get("score", 0)
        lines.append(f"  {i}. {horse} (スコア: {score:.2f})")
    return "\n".join(lines)


def format_value_bet_message(race_name: str, bets: list) -> str:
    """高期待値馬券の通知メッセージ"""
    lines = [f"\n💰 高期待値検出: {race_name}"]
    for b in bets[:5]:
        bet_type = b.get("type", "")
        combo = b.get("combo", "")
        ev = b.get("expected_value", 0)
        lines.append(f"  {bet_type} {combo} (EV: {ev:.2f})")
    return "\n".join(lines)


def format_result_message(race_name: str, results_data: list) -> str:
    """レース結果の通知メッセージ"""
    medals = ["🥇", "🥈", "🥉"]
    lines = [f"\n🏁 結果確定: {race_name}"]
    for i, r in enumerate(results_data[:3]):
        medal = medals[i] if i < 3 else ""
        horse = r.get("horse_name", "")
        lines.append(f"  {medal} {r.get('position', i+1)}着: {horse}")
    return "\n".join(lines)


def format_favorite_alert(race_name: str, matched_horses: List[str]) -> str:
    """お気に入り馬の出走通知メッセージ"""
    lines = [f"\n⭐ お気に入り馬が出走: {race_name}"]
    for h in matched_horses:
        lines.append(f"  {h}")
    return "\n".join(lines)


def format_daily_summary(summary: dict) -> str:
    """日次収支まとめの通知メッセージ"""
    o = summary.get("overview", {})
    lines = [
        f"\n📊 {date.today().isoformat()} 日次まとめ",
        f"購入: {o.get('total_bets', 0)}件",
        f"投資: ¥{o.get('total_bet_amount', 0):,}",
        f"払戻: ¥{o.get('total_payout', 0):,}",
        f"損益: ¥{o.get('total_profit', 0):+,}",
        f"回収率: {o.get('roi_percent', 0)}%",
    ]
    return "\n".join(lines)


# ============================================================
# テスト送信
# ============================================================

def test_line(token: str) -> bool:
    """LINEトークンのテスト送信"""
    return send_line(token, "\n🏇 KEIBA AI 通知テスト\nLINE通知の設定が正常に完了しました！")


def test_discord(webhook_url: str) -> bool:
    """Discord Webhookのテスト送信"""
    return send_discord(webhook_url, "🏇 KEIBA AI 通知テスト\nDiscord通知の設定が正常に完了しました！")


if __name__ == "__main__":
    print("=" * 50)
    print("通知テスト")
    print("=" * 50)
    print(f"設定ファイル: {NOTIFICATION_SETTINGS_FILE}")
    all_settings = _load_all_settings()
    print(f"登録ユーザー数: {len(all_settings)}")
    for user, s in all_settings.items():
        channels = []
        if s.get("line_token"):
            channels.append("LINE")
        if s.get("discord_webhook"):
            channels.append("Discord")
        print(f"  {user}: {', '.join(channels) if channels else '(未設定)'}")
