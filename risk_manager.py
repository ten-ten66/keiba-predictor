"""
Risk Manager - Financial Circuit Breakers
==========================================
Prevents runaway losses by enforcing daily/weekly loss limits,
consecutive-loss stops, max drawdown, and bet-size caps.

Usage:
    rm = RiskManager()
    allowed, reason = rm.check_bet_allowed(bet_amount=1000, bankroll=50000)
    if allowed:
        # place bet ...
        rm.record_result(won=False, amount=1000, payout=0)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path


class RiskManager:
    DEFAULTS = {
        "daily_loss_limit": 50000,          # yen
        "weekly_loss_limit": 150000,        # yen
        "consecutive_loss_limit": 10,       # stop after N consecutive losses
        "max_drawdown_pct": 30,             # % from peak bankroll
        "max_bet_size": 10000,              # yen per single bet
        "max_race_wager_pct": 0.10,         # max total wager per race as fraction of bankroll
        "max_bets_per_day": 50,             # cap on number of bets per day
        "cooldown_after_limit": 3600,       # seconds
    }

    def __init__(self, config=None):
        cfg = dict(self.DEFAULTS)
        if config:
            cfg.update(config)
        self.config = cfg
        self.state_file = Path("data/risk_state.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_bet_allowed(self, bet_amount, bankroll, race_total=None):
        """Returns (allowed: bool, reason: str) tuple.

        Runs through all circuit-breaker checks. Returns immediately
        on the first failure.

        Args:
            bet_amount: single-bet amount in yen being considered
            bankroll: current bankroll in yen
            race_total: if this bet is one of several on the same race,
                the total wager already placed on this race (excluding this
                bet). Used to enforce per-race aggregate cap.
        """
        state = self._load_state()

        # 0. Manual halt (kill switch)
        if state.get("halted"):
            return False, f"halted manually: {state.get('halt_reason', 'no reason given')}"

        # 1. Check cooldown
        if state.get("cooldown_until", 0) > time.time():
            remaining = int(state["cooldown_until"] - time.time())
            return False, f"cooldown active ({remaining}s remaining)"

        # 2. Check single-bet size
        if bet_amount > self.config["max_bet_size"]:
            return False, (
                f"bet_amount {bet_amount} exceeds max_bet_size "
                f"{self.config['max_bet_size']}"
            )

        # 2b. Check per-race aggregate wager
        if race_total is not None and bankroll > 0:
            combined = race_total + bet_amount
            cap = self.config["max_race_wager_pct"] * bankroll
            if combined > cap:
                return False, (
                    f"per-race wager {combined:.0f} would exceed "
                    f"{self.config['max_race_wager_pct']*100:.1f}% of bankroll ({cap:.0f})"
                )

        # 2c. Check daily bet count
        today = datetime.now().strftime("%Y-%m-%d")
        bets_today = state.get("daily", {}).get(today, {}).get("bets", 0)
        if bets_today >= self.config["max_bets_per_day"]:
            self._activate_cooldown(state)
            return False, (
                f"daily bet count limit reached: {bets_today} >= "
                f"{self.config['max_bets_per_day']}"
            )

        # 3. Check daily loss
        daily_loss = self._daily_loss(state)
        if daily_loss >= self.config["daily_loss_limit"]:
            self._activate_cooldown(state)
            return False, (
                f"daily loss limit reached: {daily_loss} >= "
                f"{self.config['daily_loss_limit']}"
            )

        # 4. Check weekly loss
        weekly_loss = self._weekly_loss(state)
        if weekly_loss >= self.config["weekly_loss_limit"]:
            self._activate_cooldown(state)
            return False, (
                f"weekly loss limit reached: {weekly_loss} >= "
                f"{self.config['weekly_loss_limit']}"
            )

        # 5. Check consecutive losses
        if state.get("consecutive_losses", 0) >= self.config["consecutive_loss_limit"]:
            self._activate_cooldown(state)
            return False, (
                f"consecutive loss limit reached: "
                f"{state['consecutive_losses']} >= "
                f"{self.config['consecutive_loss_limit']}"
            )

        # 6. Check drawdown (against peak bankroll, not peak cumulative profit)
        peak = state.get("peak_bankroll", 0)
        if peak > 0 and bankroll < peak:
            drawdown_pct = (peak - bankroll) / peak * 100
            if drawdown_pct >= self.config["max_drawdown_pct"]:
                self._activate_cooldown(state)
                return False, (
                    f"max drawdown reached: {drawdown_pct:.1f}% >= "
                    f"{self.config['max_drawdown_pct']}%"
                )

        return True, "ok"

    def halt(self, reason="manual halt"):
        """Kill switch: block all future bets until resume() is called."""
        state = self._load_state()
        state["halted"] = True
        state["halt_reason"] = str(reason)
        state["halted_at"] = datetime.now().isoformat()
        self._save_state(state)

    def resume(self):
        """Clear the halt flag."""
        state = self._load_state()
        state["halted"] = False
        state.pop("halt_reason", None)
        state.pop("halted_at", None)
        self._save_state(state)

    def record_result(self, won, amount, payout, bankroll=None):
        """Record a bet result and update state.

        Args:
            won: True if the bet was a winner.
            amount: The bet amount in yen.
            payout: The payout received (0 if lost).
            bankroll: Current bankroll after this result. Used to update
                peak_bankroll for drawdown tracking. If omitted, peak
                tracking is skipped (legacy callers).
        """
        state = self._load_state()
        today = datetime.now().strftime("%Y-%m-%d")
        profit = payout - amount

        # Update daily record
        daily = state.setdefault("daily", {})
        day_rec = daily.setdefault(today, {"bets": 0, "wagered": 0, "profit": 0, "wins": 0})
        day_rec["bets"] += 1
        day_rec["wagered"] += amount
        day_rec["profit"] += profit
        if won:
            day_rec["wins"] += 1

        # Update consecutive losses
        if won:
            state["consecutive_losses"] = 0
        else:
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1

        # Update bankroll tracking
        state["total_profit"] = state.get("total_profit", 0) + profit
        state["total_bets"] = state.get("total_bets", 0) + 1
        state["total_wagered"] = state.get("total_wagered", 0) + amount
        state["total_wins"] = state.get("total_wins", 0) + (1 if won else 0)

        # Track peak bankroll (actual wealth), not cumulative profit.
        # Drawdown is measured against peak bankroll.
        if bankroll is not None and bankroll > state.get("peak_bankroll", 0):
            state["peak_bankroll"] = bankroll

        state["last_updated"] = datetime.now().isoformat()
        self._save_state(state)

    def get_status(self):
        """Return current risk status for display/API.

        Returns:
            Dict with current risk metrics.
        """
        state = self._load_state()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = self._daily_loss(state)
        weekly_loss = self._weekly_loss(state)

        cooldown_until = state.get("cooldown_until", 0)
        cooldown_active = cooldown_until > time.time()

        bets_today = state.get("daily", {}).get(today, {}).get("bets", 0)
        return {
            "halted": bool(state.get("halted")),
            "halt_reason": state.get("halt_reason"),
            "halted_at": state.get("halted_at"),
            "daily_loss": daily_loss,
            "daily_loss_limit": self.config["daily_loss_limit"],
            "daily_loss_pct": round(
                daily_loss / self.config["daily_loss_limit"] * 100, 1
            ) if self.config["daily_loss_limit"] > 0 else 0,
            "weekly_loss": weekly_loss,
            "weekly_loss_limit": self.config["weekly_loss_limit"],
            "weekly_loss_pct": round(
                weekly_loss / self.config["weekly_loss_limit"] * 100, 1
            ) if self.config["weekly_loss_limit"] > 0 else 0,
            "consecutive_losses": state.get("consecutive_losses", 0),
            "consecutive_loss_limit": self.config["consecutive_loss_limit"],
            "bets_today": bets_today,
            "max_bets_per_day": self.config["max_bets_per_day"],
            "total_bets": state.get("total_bets", 0),
            "total_wagered": state.get("total_wagered", 0),
            "total_profit": state.get("total_profit", 0),
            "total_wins": state.get("total_wins", 0),
            "peak_bankroll": state.get("peak_bankroll", 0),
            "cooldown_active": cooldown_active,
            "cooldown_remaining_s": (
                int(cooldown_until - time.time()) if cooldown_active else 0
            ),
            "max_bet_size": self.config["max_bet_size"],
            "max_race_wager_pct": self.config["max_race_wager_pct"],
            "max_drawdown_pct": self.config["max_drawdown_pct"],
            "last_updated": state.get("last_updated"),
        }

    def reset(self):
        """Reset the risk state (admin action). Also clears halt flag."""
        self._save_state({
            "halted": False,
            "consecutive_losses": 0,
            "total_profit": 0,
            "total_bets": 0,
            "total_wagered": 0,
            "total_wins": 0,
            "peak_bankroll": 0,
            "daily": {},
            "cooldown_until": 0,
            "last_updated": datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _daily_loss(self, state):
        """Compute today's loss (positive = amount lost)."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_rec = state.get("daily", {}).get(today, {})
        profit = day_rec.get("profit", 0)
        return max(0, -profit)

    def _weekly_loss(self, state):
        """Compute this week's loss (positive = amount lost)."""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        total_loss = 0
        for day_str, day_rec in state.get("daily", {}).items():
            try:
                day_date = datetime.strptime(day_str, "%Y-%m-%d")
            except ValueError:
                continue
            if day_date >= week_start:
                profit = day_rec.get("profit", 0)
                if profit < 0:
                    total_loss += -profit
        return total_loss

    def _activate_cooldown(self, state):
        """Set cooldown timer and persist."""
        state["cooldown_until"] = time.time() + self.config["cooldown_after_limit"]
        self._save_state(state)

    def _load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "consecutive_losses": 0,
            "total_profit": 0,
            "total_bets": 0,
            "total_wagered": 0,
            "total_wins": 0,
            "peak_bankroll": 0,
            "daily": {},
            "cooldown_until": 0,
        }

    def _save_state(self, state):
        """Persist state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
