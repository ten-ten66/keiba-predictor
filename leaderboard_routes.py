"""
Leaderboard + Monte Carlo API routes
=====================================
Self-registering module: imports the FastAPI app from api_server
and registers new routes directly.

This module is imported at the bottom of api_server.py.
If that import fails or hasn't been added yet, the routes can still
be registered by running:
    python -c "import leaderboard_routes"
after api_server has been imported.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def register_routes(app, Query, HTTPException, DATA_DIR):
    """Register Monte Carlo and Leaderboard routes on the FastAPI app."""
    import pandas as pd

    # ============================================================
    # Monte Carlo Bankroll Simulation
    # ============================================================

    try:
        from monte_carlo import (
            get_strategy_simulation,
            load_results as load_mc_results,
            save_results as save_mc_results,
        )
        _monte_carlo_available = True
    except Exception:
        _monte_carlo_available = False
        logger.warning("monte_carlo module not available")

    @app.get("/api/monte-carlo")
    async def api_monte_carlo(
        strategy: str = Query(None, description="Strategy key (e.g. top_2_wide)"),
        bankroll: int = Query(100000, description="Initial bankroll"),
        bet_size: int = Query(1000, description="Bet size per race"),
    ):
        """Monte Carlo bankroll simulation for betting strategies."""
        if not _monte_carlo_available:
            raise HTTPException(status_code=503, detail="Monte Carlo module not available")

        # If no strategy specified, return all cached results (or generate them)
        if not strategy:
            cached = load_mc_results()
            if not cached:
                try:
                    cached = save_mc_results()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
            return {"status": "ok", "simulations": cached}

        # Run for specific strategy
        result = get_strategy_simulation(strategy, bankroll=bankroll, bet_size=bet_size)
        if result is None:
            cached = load_mc_results()
            available = list(cached.keys()) if cached else []
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy}' not found. Available: {available}",
            )

        return {"status": "ok", "simulation": result}

    # ============================================================
    # Prediction Leaderboard
    # ============================================================

    @app.get("/api/leaderboard")
    async def get_leaderboard():
        """Public leaderboard of prediction accuracy."""
        leaderboard = []
        _data_dir = Path(DATA_DIR)

        # --- AI predictions from honest backtest ---
        backtest_path = _data_dir / "honest_backtest_results.json"
        if backtest_path.exists():
            try:
                with open(backtest_path) as f:
                    bt = json.load(f)
                for s in bt.get("strategies", []):
                    if "Top-1" in s["strategy"]:
                        leaderboard.append({
                            "rank": 0,
                            "name": "AI (Top-1 Win)",
                            "predictions": s["total_bets"],
                            "hits": s["hit_count"],
                            "hit_rate": s["hit_rate_pct"],
                            "badge": "\U0001f916",
                        })
                        break
            except Exception:
                pass

        # --- User predictions from prediction_history/ ---
        history_dir = _data_dir / "prediction_history"
        if history_dir.exists():
            # Load race results for checking accuracy
            results_data = {}
            try:
                all_races_path = _data_dir / "_all_races.csv"
                if all_races_path.exists():
                    df_lb = pd.read_csv(str(all_races_path), dtype=str, low_memory=False)
                    if "race_id" in df_lb.columns and "\u7740\u9806" in df_lb.columns and "\u99ac\u756a" in df_lb.columns:
                        for _, row in df_lb.iterrows():
                            rid = str(row.get("race_id", ""))
                            order = str(row.get("\u7740\u9806", ""))
                            umaban = str(row.get("\u99ac\u756a", ""))
                            if rid and order == "1":
                                results_data[rid] = umaban
            except Exception:
                pass

            for hist_file in sorted(history_dir.glob("*.json")):
                username = hist_file.stem
                try:
                    with open(hist_file) as f:
                        preds = json.load(f)
                    if not isinstance(preds, list):
                        continue

                    total = 0
                    hits = 0
                    for pred in preds:
                        race_id = pred.get("race_id", "")
                        top3 = pred.get("top3", [])
                        if not race_id or not top3:
                            continue

                        winner_umaban = results_data.get(race_id)
                        if winner_umaban is None:
                            continue  # Result not available yet

                        total += 1
                        # Check if any of the user's top-3 picks won
                        picked_numbers = [str(h.get("horse_number", "")) for h in top3]
                        if winner_umaban in picked_numbers:
                            hits += 1

                    if total > 0:
                        hit_rate = round(hits / total * 100, 1)
                        leaderboard.append({
                            "rank": 0,
                            "name": username,
                            "predictions": total,
                            "hits": hits,
                            "hit_rate": hit_rate,
                            "badge": "\U0001f3c7",
                        })
                except Exception:
                    continue

        # Sort by hit_rate descending, then by predictions descending
        leaderboard.sort(key=lambda x: (-x["hit_rate"], -x["predictions"]))

        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return {
            "leaderboard": leaderboard,
            "updated_at": datetime.now().isoformat(),
        }

    logger.info("Monte Carlo + Leaderboard routes registered successfully")
