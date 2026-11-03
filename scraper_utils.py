"""
競馬予想AI - スクレイパー共通ユーティリティ
============================================
scraper.py と scraper_extra.py の共通ロジックを一元管理する。

提供機能:
  - ScraperSession : リトライ・指数バックオフ付きHTTPセッション管理
  - RateLimiter    : リクエスト間隔制御
  - parse_retry    : デコレータ形式のリトライヘルパー

使い方:
    from scraper_utils import ScraperSession

    session = ScraperSession(delay=2.0)
    soup = session.get("https://db.netkeiba.com/race/202406050811/")
    if soup:
        ...
"""

import logging
import random
import time
from functools import wraps
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------
# ロガー設定
# ----------------------------------------------------------------
logger = logging.getLogger("scraper_utils")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler()
    _ch.setFormatter(
        logging.Formatter("%(asctime)s [SCRAPER] %(levelname)s %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(_ch)

# ----------------------------------------------------------------
# 定数
# ----------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# 指数バックオフのパラメータ
_BACKOFF_BASE = 2.0      # 秒
_BACKOFF_CAP = 60.0      # 秒（上限）
_BACKOFF_JITTER = 0.3    # ランダムジッタ割合


# ----------------------------------------------------------------
# RateLimiter
# ----------------------------------------------------------------

class RateLimiter:
    """
    リクエスト間隔を一定以上保つシンプルなレート制限クラス。

    使い方:
        rl = RateLimiter(min_interval=2.0)
        rl.wait()   # 前回リクエストから min_interval 秒待つ
    """

    def __init__(self, min_interval: float = 2.0, jitter: float = 0.5):
        """
        Args:
            min_interval: リクエスト間の最小間隔（秒）
            jitter: ランダム加算の最大値（秒）。サーバー検知回避のためのノイズ
        """
        self.min_interval = min_interval
        self.jitter = jitter
        self._last_request_ts: float = 0.0

    def wait(self):
        """前回リクエストから min_interval + random(0, jitter) 秒待機する。"""
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        wait_secs = max(0.0, self.min_interval - elapsed) + random.uniform(0, self.jitter)
        if wait_secs > 0:
            time.sleep(wait_secs)
        self._last_request_ts = time.monotonic()


# ----------------------------------------------------------------
# ScraperSession
# ----------------------------------------------------------------

class ScraperSession:
    """
    リトライ・指数バックオフ・セッション管理を統合したHTTPセッション。

    scraper.py の NetkeibaScaper._get() と
    scraper_extra.py の ExtraScraper._get() の重複ロジックを統一する。

    使い方:
        session = ScraperSession(delay=2.0, max_retries=3)
        soup = session.get("https://db.netkeiba.com/race/xxx/")
        # None が返った場合はエラーまたはリトライ上限超過

        # セッションのリフレッシュ（長時間スクレイピング時に推奨）
        session.refresh()
    """

    def __init__(
        self,
        delay: float = 2.0,
        max_retries: int = 3,
        timeout: int = 30,
        headers: dict = None,
        error_threshold: int = 10,
        error_cooldown: float = 30.0,
        session_refresh_interval: int = 500,
    ):
        """
        Args:
            delay: リクエスト間の最小待機秒数（RateLimiter に渡す）
            max_retries: リトライ最大回数
            timeout: HTTP タイムアウト（秒）
            headers: カスタムHTTPヘッダー（None の場合 DEFAULT_HEADERS を使用）
            error_threshold: この件数を超えたらクールダウン
            error_cooldown: エラー多発時の追加待機秒数
            session_refresh_interval: N リクエストごとにセッションを再作成する
        """
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = headers or DEFAULT_HEADERS
        self.error_threshold = error_threshold
        self.error_cooldown = error_cooldown
        self.session_refresh_interval = session_refresh_interval

        self._rate_limiter = RateLimiter(min_interval=delay)
        self._session: Optional[requests.Session] = None
        self._request_count: int = 0
        self._error_count: int = 0
        self._consecutive_errors: int = 0

        self._create_session()

    def _create_session(self):
        """新しい requests.Session を作成する。"""
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def refresh(self):
        """既存セッションを閉じて新規作成する（長時間稼働時の接続リーク防止）。"""
        try:
            self._session.close()
        except Exception:
            pass
        self._create_session()
        logger.debug("セッションをリフレッシュしました")

    def get(
        self,
        url: str,
        encoding: str = "euc-jp",
        extra_delay: float = 0.0,
    ) -> Optional[BeautifulSoup]:
        """
        指定URLをGETし、BeautifulSoup を返す。
        失敗した場合は指数バックオフでリトライし、上限超過で None を返す。

        Args:
            url: 取得対象URL
            encoding: レスポンスのエンコーディング（デフォルト "euc-jp"）
            extra_delay: rate_limiter に加えて追加待機する秒数

        Returns:
            BeautifulSoup オブジェクト、またはエラー時は None
        """
        # エラー多発時のクールダウン
        if self._error_count >= self.error_threshold:
            logger.warning(
                f"エラー多発({self._error_count}件)。{self.error_cooldown}秒クールダウン中..."
            )
            time.sleep(self.error_cooldown)
            self._error_count = 0

        # セッション定期リフレッシュ（メモリ・接続リーク防止）
        if (self._request_count > 0
                and self._request_count % self.session_refresh_interval == 0):
            self.refresh()

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                backoff = min(
                    _BACKOFF_BASE ** attempt,
                    _BACKOFF_CAP,
                ) * (1 + random.uniform(0, _BACKOFF_JITTER))
                logger.debug(f"リトライ {attempt}/{self.max_retries}: {url} ({backoff:.1f}秒待機)")
                time.sleep(backoff)

            # レート制限
            self._rate_limiter.wait()
            if extra_delay > 0:
                time.sleep(extra_delay)

            try:
                resp = self._session.get(url, timeout=self.timeout)
                self._request_count += 1

                if resp.status_code == 200:
                    resp.encoding = encoding
                    self._consecutive_errors = 0
                    return BeautifulSoup(resp.text, "html.parser")

                if resp.status_code in (429, 503):
                    # レートリミット or サービス不可 -> 長めに待機してリトライ
                    wait = float(resp.headers.get("Retry-After", 10))
                    logger.warning(f"HTTP {resp.status_code}: {url} ({wait:.0f}秒待機後リトライ)")
                    time.sleep(wait)
                    self._error_count += 1
                    continue

                # その他の非200
                logger.warning(f"HTTP {resp.status_code}: {url}")
                self._error_count += 1
                self._consecutive_errors += 1

                # 4xx クライアントエラーはリトライしない
                if 400 <= resp.status_code < 500:
                    return None

            except requests.exceptions.ConnectionError as e:
                logger.error(f"接続エラー ({attempt+1}/{self.max_retries+1}): {e}")
                self._error_count += 1
                self._consecutive_errors += 1
                # 接続エラーはセッションリフレッシュ後にリトライ
                if attempt < self.max_retries:
                    self.refresh()

            except requests.exceptions.Timeout:
                logger.warning(f"タイムアウト ({attempt+1}/{self.max_retries+1}): {url}")
                self._error_count += 1
                self._consecutive_errors += 1

            except requests.RequestException as e:
                logger.error(f"リクエストエラー ({attempt+1}/{self.max_retries+1}): {e}")
                self._error_count += 1
                self._consecutive_errors += 1

        logger.error(f"リトライ上限到達 ({self.max_retries}回): {url}")
        return None

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def get_stats(self) -> dict:
        """取得統計を返す。"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": (
                f"{self._error_count / max(self._request_count, 1) * 100:.1f}%"
            ),
            "consecutive_errors": self._consecutive_errors,
        }


# ----------------------------------------------------------------
# デコレータ形式リトライヘルパー
# ----------------------------------------------------------------

def parse_retry(max_retries: int = 2, delay: float = 3.0):
    """
    HTMLパース処理に指数バックオフリトライを付加するデコレータ。

    関数が None または例外を返した場合にリトライする。

    使い方:
        @parse_retry(max_retries=2, delay=3.0)
        def _scrape_horse_page(self, horse_id):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    last_exc = e
                    logger.debug(
                        f"{func.__name__} 失敗 ({attempt+1}/{max_retries+1}): {e}"
                    )
                if attempt < max_retries:
                    wait = delay * (_BACKOFF_BASE ** attempt)
                    wait = min(wait + random.uniform(0, 1.0), _BACKOFF_CAP)
                    time.sleep(wait)
            if last_exc:
                logger.debug(f"{func.__name__} リトライ上限: {last_exc}")
            return None
        return wrapper
    return decorator


# ----------------------------------------------------------------
# 便利関数: 既存スクレイパーとの互換レイヤー
# ----------------------------------------------------------------

def create_default_session(delay: float = 2.0, max_retries: int = 3) -> ScraperSession:
    """デフォルト設定の ScraperSession を作成する。"""
    return ScraperSession(
        delay=delay,
        max_retries=max_retries,
        timeout=30,
        headers=DEFAULT_HEADERS,
    )
