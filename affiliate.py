"""
競馬予想AI - アフィリエイト統合フレームワーク
============================================
予測結果にアフィリエイトリンクを含め、馬券購入への導線を提供する。

リンク先:
  - JRA IPAT (公式オンライン馬券購入)
  - 楽天競馬
"""

import os
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("KEIBA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

# ============================================================
# アフィリエイトリンク定義
# ============================================================

AFFILIATE_LINKS = {
    "jra_ipat": {
        "url": "https://www.jra.go.jp/dento/member/",
        "description": "JRA IPAT 会員登録",
        "cta": "\u99ac\u5238\u3092\u8cfc\u5165\u3059\u308b \u2192",
        "provider": "JRA",
        "icon": "\U0001f3c7",
    },
    "rakuten_keiba": {
        "url": "https://keiba.rakuten.co.jp/",
        "description": "\u697d\u5929\u7af6\u99ac",
        "cta": "\u697d\u5929\u7af6\u99ac\u3067\u8cfc\u5165 \u2192",
        "provider": "\u697d\u5929",
        "icon": "\U0001f4b0",
    },
}


def get_affiliate_link(provider: str = "jra_ipat") -> dict:
    """
    指定プロバイダーのアフィリエイトリンクを取得する。

    Args:
        provider: "jra_ipat" or "rakuten_keiba"

    Returns:
        {"url": "...", "description": "...", "cta": "...", ...}
    """
    return AFFILIATE_LINKS.get(provider, {})


def get_all_affiliate_links() -> dict:
    """全アフィリエイトリンクを返す"""
    return AFFILIATE_LINKS


def get_affiliate_cta_for_prediction(has_value_bets: bool = False) -> dict:
    """
    予測レスポンスに含めるアフィリエイトCTA情報を生成する。

    Args:
        has_value_bets: バリューベットがあるか（CTAの強調度合いを変える）

    Returns:
        {
            "primary": {"url": ..., "cta": ..., "description": ...},
            "secondary": {"url": ..., "cta": ..., "description": ...},
            "message": "..."
        }
    """
    primary = get_affiliate_link("jra_ipat")
    secondary = get_affiliate_link("rakuten_keiba")

    if has_value_bets:
        message = "\U0001f525 \u30d0\u30ea\u30e5\u30fc\u30d9\u30c3\u30c8\u691c\u51fa\uff01\u4eca\u3059\u3050\u99ac\u5238\u3092\u8cfc\u5165\u3057\u307e\u3057\u3087\u3046"
    else:
        message = "\u6c17\u306b\u306a\u308b\u99ac\u304c\u898b\u3064\u304b\u3063\u305f\u3089\u3001\u99ac\u5238\u3092\u8cfc\u5165\u3057\u307e\u3057\u3087\u3046"

    return {
        "primary": primary,
        "secondary": secondary,
        "message": message,
    }
