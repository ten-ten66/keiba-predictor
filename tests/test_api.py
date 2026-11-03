"""
API Test Suite for KEIBA AI Predictor
======================================
Run with: pytest tests/test_api.py -v

Requires the API server to be running at http://127.0.0.1:8000
(override with KEIBA_TEST_BASE, e.g. KEIBA_TEST_BASE=http://127.0.0.1:8002)
"""

import os

import pytest
import requests
import time

BASE = os.environ.get("KEIBA_TEST_BASE", "http://127.0.0.1:8000")
_token_cache = {}


def get_token():
    """Obtain a JWT token for admin user, cached for the session."""
    if "token" in _token_cache:
        return _token_cache["token"]
    r = requests.post(
        f"{BASE}/api/auth/login",
        json={"username": "admin", "password": os.environ["KEIBA_ADMIN_PASSWORD"]},
        timeout=10,
    )
    assert r.status_code == 200, f"Login failed: {r.status_code} {r.text}"
    token = r.json()["access_token"]
    _token_cache["token"] = token
    return token


def auth_headers():
    return {"Authorization": f"Bearer {get_token()}"}


# ============================================================
# 1. Health check (public)
# ============================================================

class TestHealth:
    def test_health_endpoint(self):
        r = requests.get(f"{BASE}/api/health", timeout=10)
        assert r.status_code in (200, 503)
        data = r.json()
        assert data["status"] in ("ok", "loading")

    def test_health_has_timestamp(self):
        r = requests.get(f"{BASE}/api/health", timeout=10)
        if r.status_code == 200:
            assert "timestamp" in r.json()


# ============================================================
# 2. Authentication
# ============================================================

class TestAuth:
    def test_login_success(self):
        r = requests.post(
            f"{BASE}/api/auth/login",
            json={"username": "admin", "password": os.environ["KEIBA_ADMIN_PASSWORD"]},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self):
        r = requests.post(
            f"{BASE}/api/auth/login",
            json={"username": "admin", "password": "wrongpassword123"},
            timeout=10,
        )
        assert r.status_code == 401

    def test_auth_me(self):
        r = requests.get(f"{BASE}/api/auth/me", headers=auth_headers(), timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["user"]["username"] == "admin"

    def test_auth_me_no_token(self):
        r = requests.get(f"{BASE}/api/auth/me", timeout=10)
        assert r.status_code in (401, 403)


# ============================================================
# 3. Status (authenticated)
# ============================================================

class TestStatus:
    def test_status_authenticated(self):
        r = requests.get(f"{BASE}/api/status", headers=auth_headers(), timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_status_unauthenticated(self):
        r = requests.get(f"{BASE}/api/status", timeout=10)
        assert r.status_code in (401, 403)


# ============================================================
# 4. Races
# ============================================================

class TestRaces:
    def test_races_upcoming(self):
        r = requests.get(
            f"{BASE}/api/races/upcoming",
            params={"date": "20260321"},
            headers=auth_headers(),
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "races" in data or "data" in data

    def test_races_today(self):
        r = requests.get(f"{BASE}/api/races/today", timeout=10)
        assert r.status_code == 200


# ============================================================
# 5. Best bets (public)
# ============================================================

class TestBestBets:
    def test_best_bets(self):
        r = requests.get(
            f"{BASE}/api/best-bets",
            params={"date": "20260321"},
            timeout=15,
        )
        assert r.status_code == 200


# ============================================================
# 6. Performance (public)
# ============================================================

class TestPerformance:
    def test_performance(self):
        r = requests.get(f"{BASE}/api/performance", timeout=10)
        assert r.status_code == 200

    def test_performance_weekly(self):
        r = requests.get(f"{BASE}/api/performance/weekly", timeout=10)
        assert r.status_code == 200


# ============================================================
# 7. Search
# ============================================================

class TestSearch:
    def test_search(self):
        r = requests.get(
            f"{BASE}/api/search",
            params={"q": "東京"},
            timeout=10,
        )
        assert r.status_code == 200


# ============================================================
# 8. Jockey / Trainer Stats (B2)
# ============================================================

class TestJockeyTrainerStats:
    def test_jockey_ranking(self):
        r = requests.get(
            f"{BASE}/api/jockey/ranking",
            params={"min_rides": 100, "limit": 10},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "ranking" in data
        assert "total" in data

    def test_trainer_ranking(self):
        r = requests.get(
            f"{BASE}/api/trainer/ranking",
            params={"min_rides": 50, "limit": 10},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "ranking" in data

    def test_jockey_stats_not_found(self):
        r = requests.get(f"{BASE}/api/jockey/nonexistent_99999/stats", timeout=10)
        assert r.status_code == 404

    def test_trainer_stats_not_found(self):
        r = requests.get(f"{BASE}/api/trainer/nonexistent_99999/stats", timeout=10)
        assert r.status_code == 404

    def test_jockey_ranking_structure(self):
        """Verify ranking entries have expected fields."""
        r = requests.get(
            f"{BASE}/api/jockey/ranking",
            params={"min_rides": 100, "limit": 3},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        if data["ranking"]:
            entry = data["ranking"][0]
            assert "jockey_id" in entry
            assert "jockey_name" in entry
            assert "win_rate" in entry
            assert "total_rides" in entry
            assert entry["total_rides"] >= 100


# ============================================================
# 9. Admin endpoints
# ============================================================

class TestAdmin:
    def test_admin_health_check(self):
        r = requests.get(
            f"{BASE}/api/admin/health-check",
            headers=auth_headers(),
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "overall" in data
        assert data["overall"] in ("ok", "warning", "error")
        assert "checks" in data

    def test_admin_health_check_unauthenticated(self):
        r = requests.get(f"{BASE}/api/admin/health-check", timeout=10)
        assert r.status_code in (401, 403)

    def test_admin_users(self):
        r = requests.get(
            f"{BASE}/api/admin/users",
            headers=auth_headers(),
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        users = data.get("users", [])
        assert isinstance(users, list)
        assert any(u["username"] == "admin" for u in users)

    def test_admin_system(self):
        r = requests.get(
            f"{BASE}/api/admin/system",
            headers=auth_headers(),
            timeout=10,
        )
        assert r.status_code == 200


# ============================================================
# 10. Balance / History
# ============================================================

class TestBalance:
    def test_balance(self):
        r = requests.get(f"{BASE}/api/balance", timeout=10)
        assert r.status_code == 200

    def test_history(self):
        r = requests.get(
            f"{BASE}/api/history",
            params={"limit": 5},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "bets" in data


# ============================================================
# 11. Venues
# ============================================================

class TestVenues:
    def test_venues_list(self):
        r = requests.get(f"{BASE}/api/venues", timeout=10)
        assert r.status_code == 200


# ============================================================
# 12. Rate limiting sanity check
# ============================================================

class TestRateLimiting:
    def test_normal_request_not_throttled(self):
        """A single request should not be rate limited."""
        r = requests.get(f"{BASE}/api/health", timeout=10)
        assert r.status_code != 429
