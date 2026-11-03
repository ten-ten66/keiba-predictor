"""
Automated Betting Framework (Paper Trading Only)
=================================================
Framework for automated bet evaluation and execution in paper-trading mode.

IMPORTANT: mode="live" is NOT implemented and should NEVER be implemented
without explicit user consent and proper regulatory review.
This module is for PAPER TRADING ONLY.

Components:
  AutoBettor    - Main orchestrator: evaluate races, place paper trades
  RiskManager   - Financial circuit breakers (imported from risk_manager.py)
  Kelly sizing  - Bet sizing via fractional Kelly criterion

API Endpoints:
  GET  /api/auto-bet/status        Current auto-bettor state (paper P&L, active bets)
  POST /api/admin/auto-bet/enable  Start paper trading
  POST /api/admin/auto-bet/disable Stop paper trading
  GET  /api/auto-bet/trades        Paper trade history

Usage:
    bettor = AutoBettor(strategy="ev_1.3", mode="paper")
    bets = bettor.evaluate_race(predictions, odds, race_info)
    if bets:
        bettor.execute_paper_trade(bets, race_id)
    # Later, after race completes:
    bettor.settle_paper_trade(race_id, results)
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
TRADES_FILE = DATA_DIR / "paper_trades.json"
STATE_FILE = DATA_DIR / "auto_bettor_state.json"


# ============================================================
# Auto Bettor
# ============================================================

class AutoBettor:
    """
    Automated betting framework for paper trading.

    Evaluates races, finds value bets, sizes positions using Kelly criterion,
    and tracks simulated P&L.

    PAPER TRADING ONLY. Live betting is not supported.
    """

    # Strategy presets: name -> EV threshold
    STRATEGIES = {
        "ev_1.15": 1.15,
        "ev_1.2": 1.2,
        "ev_1.3": 1.3,
        "ev_1.5": 1.5,
        "conservative": 1.5,
        "moderate": 1.3,
        "aggressive": 1.15,
    }

    # Named Kelly fractions. Half-Kelly is a common compromise; eighth is for
    # conservative / high-variance strategies.
    KELLY_PRESETS = {
        "full": 1.0,
        "half": 0.5,
        "quarter": 0.25,
        "eighth": 0.125,
    }

    def __init__(
        self,
        strategy: str = "ev_1.3",
        mode: str = "paper",
        bankroll: int = 100000,
        kelly_fraction=0.25,
        min_edge_score: float = 50,
        max_bet_pct: float = 0.05,
        min_bet: int = 100,
        uncertainty_discount: bool = True,
    ):
        """
        Args:
            strategy: Betting strategy name (see STRATEGIES)
            mode: Must be "paper". Live trading is NOT supported.
            bankroll: Starting simulated bankroll in yen
            kelly_fraction: Fraction of full Kelly to use. Accepts a float
                (e.g. 0.25) or a preset string ("full"/"half"/"quarter"/"eighth").
            min_edge_score: Minimum race edge score (0-100) to bet
            max_bet_pct: Maximum bet as fraction of bankroll
            min_bet: Minimum bet size in yen
            uncertainty_discount: If True and predictions carry ev_std, size
                bets on the lower confidence bound instead of the mean, scaling
                Kelly down by up to 70% when relative std is large.
        """
        if isinstance(kelly_fraction, str):
            if kelly_fraction not in self.KELLY_PRESETS:
                raise ValueError(
                    f"Unknown kelly preset {kelly_fraction!r}. "
                    f"Valid: {list(self.KELLY_PRESETS)} or a float."
                )
            kelly_fraction = self.KELLY_PRESETS[kelly_fraction]
        if mode != "paper":
            raise ValueError(
                "Only mode='paper' is supported. Live betting requires "
                "explicit user consent and regulatory review."
            )

        self.strategy = strategy
        self.mode = mode
        self.ev_threshold = self.STRATEGIES.get(strategy, 1.3)
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.uncertainty_discount = uncertainty_discount
        self.min_edge_score = min_edge_score
        self.max_bet_pct = max_bet_pct
        self.min_bet = min_bet
        self.enabled = False

        # Import risk manager
        from risk_manager import RiskManager
        self.risk_manager = RiskManager()

        # Load state
        self.paper_trades: List[dict] = []
        self._load_state()

    # ------------------------------------------------------------------
    # Core Logic
    # ------------------------------------------------------------------

    def evaluate_race(
        self,
        predictions: List[Dict],
        odds: Dict[int, float],
        race_info: Dict,
    ) -> Optional[List[Dict]]:
        """
        Decide whether to bet on this race and how much.

        Args:
            predictions: List of dicts with keys: horse_number, horse_name,
                        prob_win, prob_top3, composite_score
            odds: Dict mapping horse_number -> current odds
            race_info: Dict with course_type, distance, venue, race_name,
                      n_runners, etc.

        Returns:
            List of bet dicts, or None if no bets.
        """
        if not self.enabled:
            return None

        # 1. Check risk limits
        allowed, reason = self.risk_manager.check_bet_allowed(
            self.min_bet, self.bankroll
        )
        if not allowed:
            logger.info("Risk manager blocked bet: %s", reason)
            return None

        # 2. Check race selection model (edge score)
        edge_score = self._get_edge_score(race_info)
        if edge_score < self.min_edge_score:
            logger.debug(
                "Race %s edge score %.1f below threshold %.1f",
                race_info.get("race_id", "?"), edge_score, self.min_edge_score,
            )
            return None

        # 3. Find value bets using overlay model or simple EV calc
        value_bets = self._find_value_bets(predictions, odds)

        # 4. Filter by EV threshold
        bets = [b for b in value_bets if b["ev"] >= self.ev_threshold]

        if not bets:
            return None

        # 5. Size bets using Kelly criterion
        for bet in bets:
            bet["amount"] = self._kelly_size(bet)
            bet["edge_score"] = edge_score

        # Sort by EV descending
        bets.sort(key=lambda x: x["ev"], reverse=True)

        # Limit to top 3 bets per race to manage risk
        top_bets = bets[:3]

        # Enforce per-bet / per-race risk caps; drop bets that exceed limits
        allowed_bets = []
        race_total = 0
        for b in top_bets:
            ok, reason = self.risk_manager.check_bet_allowed(
                bet_amount=b["amount"],
                bankroll=self.bankroll,
                race_total=race_total,
            )
            if not ok:
                logger.info(
                    "Risk manager dropped bet on horse %s (amount=%d): %s",
                    b.get("horse_number", "?"), b["amount"], reason,
                )
                continue
            allowed_bets.append(b)
            race_total += b["amount"]

        return allowed_bets or None

    def execute_paper_trade(self, bets: List[Dict], race_id: str, race_info: Dict = None):
        """
        Record a paper trade (simulated bet execution).

        Args:
            bets: List of bet dicts from evaluate_race()
            race_id: The race identifier
            race_info: Optional race metadata
        """
        if self.mode != "paper":
            raise RuntimeError("Only paper trading is supported")

        total_wagered = sum(b["amount"] for b in bets)

        trade = {
            "trade_id": f"PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{race_id[-4:]}",
            "race_id": race_id,
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy,
            "ev_threshold": self.ev_threshold,
            "bankroll_at_entry": self.bankroll,
            "bets": bets,
            "total_wagered": total_wagered,
            "status": "pending",  # pending -> settled
            "race_info": {
                "race_name": race_info.get("race_name", "") if race_info else "",
                "venue": race_info.get("venue", "") if race_info else "",
                "distance": race_info.get("distance", 0) if race_info else 0,
            },
            "result": None,
        }

        self.paper_trades.append(trade)
        self._save_state()

        logger.info(
            "Paper trade placed: %s, %d bets, total wagered: %d yen",
            race_id, len(bets), total_wagered,
        )

    def settle_paper_trade(self, race_id: str, results: Dict[int, int]):
        """
        Settle a paper trade with actual race results.

        Args:
            race_id: The race identifier
            results: Dict mapping horse_number -> finish_position
        """
        for trade in self.paper_trades:
            if trade["race_id"] == race_id and trade["status"] == "pending":
                total_payout = 0
                total_wagered = 0

                for bet in trade["bets"]:
                    hn = bet["horse_number"]
                    finish = results.get(hn, 99)
                    bet["finish_position"] = finish
                    bet["won"] = (finish == 1)

                    amount = bet["amount"]
                    total_wagered += amount

                    if bet["won"]:
                        payout = amount * bet["odds"]
                        total_payout += payout
                        bet["payout"] = round(payout)
                        # Record win in risk manager
                        self.risk_manager.record_result(
                            won=True, amount=amount, payout=payout,
                            bankroll=self.bankroll + (payout - amount),
                        )
                    else:
                        bet["payout"] = 0
                        self.risk_manager.record_result(
                            won=False, amount=amount, payout=0,
                            bankroll=self.bankroll - amount,
                        )

                profit = total_payout - total_wagered
                self.bankroll += profit

                trade["status"] = "settled"
                trade["settled_at"] = datetime.now().isoformat()
                trade["result"] = {
                    "total_wagered": total_wagered,
                    "total_payout": round(total_payout),
                    "profit": round(profit),
                    "bankroll_after": round(self.bankroll),
                }

                self._save_state()
                logger.info(
                    "Paper trade settled: %s, P&L: %+d yen, bankroll: %d",
                    race_id, profit, self.bankroll,
                )
                return

        logger.warning("No pending trade found for race %s", race_id)

    # ------------------------------------------------------------------
    # Status & Reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current auto-bettor state for API/dashboard."""
        settled = [t for t in self.paper_trades if t["status"] == "settled"]
        pending = [t for t in self.paper_trades if t["status"] == "pending"]

        total_wagered = sum(
            t["result"]["total_wagered"] for t in settled if t.get("result")
        )
        total_payout = sum(
            t["result"]["total_payout"] for t in settled if t.get("result")
        )
        total_profit = total_payout - total_wagered

        total_bets = sum(len(t["bets"]) for t in settled)
        winning_bets = sum(
            sum(1 for b in t["bets"] if b.get("won", False))
            for t in settled
        )

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "strategy": self.strategy,
            "ev_threshold": self.ev_threshold,
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.bankroll),
            "paper_pnl": round(total_profit),
            "paper_roi_pct": round(
                total_profit / total_wagered * 100, 1
            ) if total_wagered > 0 else 0,
            "total_trades": len(self.paper_trades),
            "settled_trades": len(settled),
            "pending_trades": len(pending),
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "win_rate": round(
                winning_bets / total_bets * 100, 1
            ) if total_bets > 0 else 0,
            "total_wagered": round(total_wagered),
            "total_payout": round(total_payout),
            "risk_status": self.risk_manager.get_status(),
        }

    def get_trades(self, limit: int = 50, status: str = None) -> List[Dict]:
        """Return paper trade history, newest first."""
        trades = self.paper_trades
        if status:
            trades = [t for t in trades if t["status"] == status]
        return list(reversed(trades[-limit:]))

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_edge_score(self, race_info: Dict) -> float:
        """Get edge score for a race using race_selector."""
        try:
            from race_selector import get_edge_score_for_race
            return get_edge_score_for_race(race_info)
        except Exception as e:
            logger.debug("Edge score computation failed: %s", e)
            return 50.0  # neutral score

    def _find_value_bets(
        self, predictions: List[Dict], odds: Dict[int, float]
    ) -> List[Dict]:
        """
        Find value bets by comparing model probabilities with market odds.

        Tries the overlay model first, falls back to simple EV calculation.
        """
        # Try overlay model
        try:
            from odds_overlay import OddsOverlayModel
            overlay_path = BASE_DIR / "models" / "odds_overlay.pkl"
            if overlay_path.exists():
                model = OddsOverlayModel.load(str(overlay_path))
                results = model.predict_from_base_predictions(predictions, odds)
                return results
        except Exception as e:
            logger.debug("Overlay model failed, using simple EV: %s", e)

        # Fallback: simple EV calculation
        results = []
        for pred in predictions:
            hn = int(pred.get("horse_number", 0))
            horse_odds = odds.get(hn, 0)
            if horse_odds <= 1:
                continue

            prob_win = pred.get("prob_win", 0)
            implied_prob = 1.0 / horse_odds
            ev = prob_win * horse_odds
            edge = prob_win - implied_prob

            # Kelly
            b = horse_odds - 1
            kelly = max(0, (b * prob_win - (1 - prob_win)) / b) if b > 0 else 0

            if ev > 1.3:
                bet_type = "STRONG_VALUE"
            elif ev > 1.15:
                bet_type = "VALUE"
            elif ev > 1.0:
                bet_type = "MARGINAL"
            else:
                bet_type = "PASS"

            results.append({
                "horse_number": hn,
                "horse_name": pred.get("horse_name", ""),
                "odds": horse_odds,
                "model_prob": round(prob_win, 4),
                "market_prob": round(implied_prob, 4),
                "ev": round(ev, 3),
                "edge": round(edge, 4),
                "kelly_fraction": round(kelly, 4),
                "ev_std": float(pred.get("ev_std", 0) or 0),
                "ev_mean": float(pred.get("ev_mean", 0) or 0),
                "bet_type": bet_type,
                "is_value_bet": ev >= self.ev_threshold,
            })

        return results

    def _kelly_size(self, bet: Dict) -> int:
        """
        Size a bet using fractional Kelly criterion.

        When uncertainty_discount is enabled and the bet carries ev_std > 0,
        Kelly is scaled down by a factor between 0.3 and 1.0 based on the
        relative std (std/mean). High uncertainty → smaller bet.

        Returns bet amount in yen (integer).
        """
        kelly = bet.get("kelly_fraction", 0)
        if kelly <= 0:
            return self.min_bet

        # Uncertainty-aware discount
        if self.uncertainty_discount:
            ev_std = float(bet.get("ev_std", 0) or 0)
            ev_mean = float(bet.get("ev_mean", 0) or bet.get("model_prob", 0) or 0)
            if ev_std > 0 and ev_mean > 0:
                rel = ev_std / ev_mean  # coefficient of variation
                discount = max(0.3, min(1.0, 1.0 - rel))
                kelly *= discount

        # Apply Kelly fraction (e.g., quarter-Kelly)
        fraction = kelly * self.kelly_fraction

        # Cap at max_bet_pct of bankroll
        fraction = min(fraction, self.max_bet_pct)

        amount = self.bankroll * fraction
        amount = max(amount, self.min_bet)
        amount = min(amount, self.bankroll * self.max_bet_pct)

        # Round to nearest 100 yen
        return max(self.min_bet, int(round(amount / 100) * 100))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load auto-bettor state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.enabled = state.get("enabled", False)
                self.bankroll = state.get("bankroll", self.initial_bankroll)
                self.strategy = state.get("strategy", self.strategy)
                self.ev_threshold = self.STRATEGIES.get(
                    self.strategy, state.get("ev_threshold", 1.3)
                )
            except (json.JSONDecodeError, IOError):
                pass

        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE, "r", encoding="utf-8") as f:
                    self.paper_trades = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.paper_trades = []

    def _save_state(self):
        """Persist auto-bettor state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        state = {
            "enabled": self.enabled,
            "mode": self.mode,
            "strategy": self.strategy,
            "ev_threshold": self.ev_threshold,
            "bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "last_updated": datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.paper_trades, f, ensure_ascii=False, indent=2)


# ============================================================
# Singleton instance
# ============================================================

_auto_bettor: Optional[AutoBettor] = None


def get_auto_bettor() -> AutoBettor:
    """Get or create the singleton AutoBettor instance."""
    global _auto_bettor
    if _auto_bettor is None:
        _auto_bettor = AutoBettor(strategy="ev_1.3", mode="paper")
    return _auto_bettor


# ============================================================
# API Router
# ============================================================

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel

    auto_bet_router = APIRouter(tags=["Auto Betting (Paper)"])

    @auto_bet_router.get("/api/auto-bet/status")
    async def api_auto_bet_status():
        """Get current auto-bettor state: paper P&L, active bets."""
        bettor = get_auto_bettor()
        return {"status": "ok", **bettor.get_status()}

    @auto_bet_router.get("/api/auto-bet/trades")
    async def api_auto_bet_trades(
        limit: int = Query(50, ge=1, le=500),
        trade_status: Optional[str] = Query(None, alias="status"),
    ):
        """Get paper trade history."""
        bettor = get_auto_bettor()
        trades = bettor.get_trades(limit=limit, status=trade_status)
        return {
            "status": "ok",
            "count": len(trades),
            "trades": trades,
        }

    class EnableAutoBetRequest(BaseModel):
        strategy: str = "ev_1.3"
        bankroll: int = 100000
        kelly_fraction: float = 0.25

    @auto_bet_router.post("/api/admin/auto-bet/enable")
    async def api_auto_bet_enable(req: EnableAutoBetRequest = None):
        """Start paper trading."""
        global _auto_bettor
        if req is None:
            req = EnableAutoBetRequest()

        bettor = get_auto_bettor()

        if req.strategy not in AutoBettor.STRATEGIES:
            raise HTTPException(
                400,
                f"Unknown strategy: {req.strategy}. "
                f"Available: {list(AutoBettor.STRATEGIES.keys())}",
            )

        bettor.strategy = req.strategy
        bettor.ev_threshold = AutoBettor.STRATEGIES[req.strategy]
        bettor.kelly_fraction = req.kelly_fraction
        if not bettor.paper_trades:
            bettor.bankroll = req.bankroll
            bettor.initial_bankroll = req.bankroll
        bettor.enabled = True
        bettor._save_state()

        return {
            "status": "ok",
            "message": "Paper trading enabled",
            "strategy": bettor.strategy,
            "ev_threshold": bettor.ev_threshold,
            "bankroll": bettor.bankroll,
            "mode": "paper",
            "warning": "This is PAPER TRADING only. No real bets are placed.",
        }

    @auto_bet_router.post("/api/admin/auto-bet/disable")
    async def api_auto_bet_disable():
        """Stop paper trading."""
        bettor = get_auto_bettor()
        bettor.enabled = False
        bettor._save_state()

        return {
            "status": "ok",
            "message": "Paper trading disabled",
            **bettor.get_status(),
        }

except ImportError:
    auto_bet_router = None
    logger.warning("FastAPI not available; auto_bet_router disabled")


# ============================================================
# CLI / Testing
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Auto Bettor - Paper Trading Test")
    print("=" * 60)

    bettor = AutoBettor(strategy="ev_1.3", mode="paper", bankroll=100000)
    bettor.enabled = True

    # Mock predictions
    predictions = [
        {"horse_number": 1, "horse_name": "Thunder Bolt", "prob_win": 0.25, "prob_top3": 0.55},
        {"horse_number": 2, "horse_name": "Star Runner", "prob_win": 0.18, "prob_top3": 0.45},
        {"horse_number": 3, "horse_name": "Golden Arrow", "prob_win": 0.12, "prob_top3": 0.35},
        {"horse_number": 4, "horse_name": "Lucky Day", "prob_win": 0.08, "prob_top3": 0.28},
    ]
    odds = {1: 3.5, 2: 5.0, 3: 12.0, 4: 25.0}

    race_info = {
        "race_id": "202603200101",
        "race_name": "Test Race",
        "venue": "Tokyo",
        "distance": 1600,
        "course_type": "turf",
        "n_runners": 12,
    }

    print(f"\nStrategy: {bettor.strategy} (EV threshold: {bettor.ev_threshold})")
    print(f"Bankroll: {bettor.bankroll:,} yen")

    # Evaluate
    bets = bettor.evaluate_race(predictions, odds, race_info)
    if bets:
        print(f"\nFound {len(bets)} value bets:")
        for b in bets:
            print(f"  #{b['horse_number']} {b['horse_name']}: "
                  f"EV={b['ev']:.3f}, odds={b['odds']}, "
                  f"kelly={b.get('kelly_fraction', 0):.4f}, "
                  f"amount={b['amount']} yen")

        # Execute paper trade
        bettor.execute_paper_trade(bets, "202603200101", race_info)
        print(f"\nPaper trade placed.")

        # Settle (horse #3 wins)
        results = {1: 3, 2: 2, 3: 1, 4: 7}
        bettor.settle_paper_trade("202603200101", results)

        # Status
        status = bettor.get_status()
        print(f"\nPaper P&L: {status['paper_pnl']:+,} yen")
        print(f"Bankroll: {status['current_bankroll']:,} yen")
        print(f"Win rate: {status['win_rate']}%")
    else:
        print("\nNo value bets found for this race.")

    print("\nDone.")
