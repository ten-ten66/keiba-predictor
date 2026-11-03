"""
Monte Carlo Bankroll Simulation
================================
Bootstrap simulation of bankroll evolution for each betting strategy.
Reads honest_backtest_results.json and generates probabilistic outcomes.
"""

import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def simulate_bankroll(strategy_results, n_simulations=10000, initial_bankroll=100000, bet_size=1000):
    """
    Monte Carlo simulation of bankroll evolution.

    strategy_results: list of (won: bool, payout: float) tuples from backtest
    Returns: simulation statistics
    """
    n_bets = len(strategy_results)
    if n_bets == 0:
        return {
            "n_simulations": n_simulations,
            "initial_bankroll": initial_bankroll,
            "bet_size": bet_size,
            "n_bets_per_sim": 0,
            "mean_final": float(initial_bankroll),
            "median_final": float(initial_bankroll),
            "std_final": 0.0,
            "percentile_5": float(initial_bankroll),
            "percentile_25": float(initial_bankroll),
            "percentile_75": float(initial_bankroll),
            "percentile_95": float(initial_bankroll),
            "ruin_probability": 0.0,
            "profit_probability": 0.0,
            "mean_max_drawdown": 0.0,
            "worst_drawdown": 0.0,
            "optimal_bet_size": bet_size,
        }

    # Convert to numpy arrays
    outcomes = np.array([(1 if r[0] else 0, r[1]) for r in strategy_results])

    final_bankrolls = []
    max_drawdowns = []
    ruin_count = 0  # bankroll hits 0

    for _ in range(n_simulations):
        # Randomly shuffle bet order (bootstrap)
        indices = np.random.choice(n_bets, n_bets, replace=True)

        bankroll = initial_bankroll
        peak = bankroll
        max_dd = 0

        for idx in indices:
            won, payout = outcomes[idx]
            if won:
                bankroll += payout - bet_size
            else:
                bankroll -= bet_size

            if bankroll <= 0:
                ruin_count += 1
                break

            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak
            max_dd = max(max_dd, dd)

        final_bankrolls.append(bankroll)
        max_drawdowns.append(max_dd)

    final_bankrolls = np.array(final_bankrolls)

    return {
        "n_simulations": n_simulations,
        "initial_bankroll": initial_bankroll,
        "bet_size": bet_size,
        "n_bets_per_sim": n_bets,
        "mean_final": float(np.mean(final_bankrolls)),
        "median_final": float(np.median(final_bankrolls)),
        "std_final": float(np.std(final_bankrolls)),
        "percentile_5": float(np.percentile(final_bankrolls, 5)),
        "percentile_25": float(np.percentile(final_bankrolls, 25)),
        "percentile_75": float(np.percentile(final_bankrolls, 75)),
        "percentile_95": float(np.percentile(final_bankrolls, 95)),
        "ruin_probability": ruin_count / n_simulations,
        "profit_probability": float(np.mean(final_bankrolls > initial_bankroll)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "worst_drawdown": float(np.max(max_drawdowns)),
        "optimal_bet_size": _find_optimal_bet_size(outcomes, initial_bankroll),
    }


def _find_optimal_bet_size(outcomes, bankroll, sizes=[100, 200, 500, 1000, 2000, 5000]):
    """Find bet size that maximizes expected bankroll while keeping ruin < 5%."""
    best = {"size": 100, "expected": bankroll}
    for size in sizes:
        sim = simulate_bankroll(
            [(bool(o[0]), float(o[1])) for o in outcomes],
            n_simulations=1000, initial_bankroll=bankroll, bet_size=size
        )
        if sim["ruin_probability"] < 0.05 and sim["mean_final"] > best["expected"]:
            best = {"size": size, "expected": sim["mean_final"]}
    return best["size"]


def _synthesize_bet_outcomes(strategy_info, bet_amount=100):
    """
    Synthesize individual bet outcomes from aggregate strategy statistics.

    Since honest_backtest_results.json only has summary stats (hit_rate, total_bets,
    total_return), we reconstruct plausible individual outcomes.
    """
    total_bets = strategy_info["total_bets"]
    hit_count = strategy_info["hit_count"]
    total_return = strategy_info["total_return"]
    total_bet_amount = strategy_info["total_bet_amount"]

    # Average payout per winning bet
    if hit_count > 0:
        avg_payout_per_win = total_return / hit_count
    else:
        avg_payout_per_win = 0

    # Scale payouts proportionally to the bet_size we use in simulation
    scale = bet_amount / (total_bet_amount / total_bets) if total_bets > 0 else 1.0

    results = []
    for i in range(total_bets):
        if i < hit_count:
            # Winning bet: add some variance around average payout
            payout = avg_payout_per_win * scale
            results.append((True, payout))
        else:
            results.append((False, 0.0))

    return results


def _strategy_key(name):
    """Convert strategy name to a URL-safe key."""
    return (name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace(">", "gt")
            .replace(".", "")
            .replace("ワイド", "wide"))


def run_all_simulations():
    """Run Monte Carlo simulations for all strategies in honest_backtest_results.json."""
    backtest_path = DATA_DIR / "honest_backtest_results.json"
    if not backtest_path.exists():
        logger.warning("honest_backtest_results.json not found")
        return {}

    with open(backtest_path) as f:
        backtest = json.load(f)

    strategies = backtest.get("strategies", [])
    results = {}

    for strat in strategies:
        name = strat["strategy"]
        key = _strategy_key(name)
        logger.info(f"Running Monte Carlo for: {name} ({key})")

        # Synthesize individual bet outcomes from aggregate stats
        bet_outcomes = _synthesize_bet_outcomes(strat, bet_amount=100)

        # Run simulation with default parameters
        sim = simulate_bankroll(
            bet_outcomes,
            n_simulations=10000,
            initial_bankroll=100000,
            bet_size=1000,
        )
        sim["strategy_name"] = name
        sim["strategy_key"] = key
        sim["source_hit_rate"] = strat["hit_rate_pct"]
        sim["source_roi"] = strat["roi_pct"]

        results[key] = sim

    return results


def save_results(results=None):
    """Run simulations and save to data/monte_carlo_results.json."""
    if results is None:
        results = run_all_simulations()

    out_path = DATA_DIR / "monte_carlo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Monte Carlo results saved to {out_path}")
    return results


def load_results():
    """Load cached Monte Carlo results."""
    path = DATA_DIR / "monte_carlo_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_strategy_simulation(strategy_key, bankroll=100000, bet_size=1000):
    """
    Get simulation for a specific strategy, optionally with custom bankroll/bet_size.
    If custom parameters differ from defaults, re-run simulation.
    """
    cached = load_results()

    # If default params and cached results exist, return cached
    if cached and strategy_key in cached:
        cached_result = cached[strategy_key]
        if bankroll == 100000 and bet_size == 1000:
            return cached_result

    # Need to run with custom params
    backtest_path = DATA_DIR / "honest_backtest_results.json"
    if not backtest_path.exists():
        return None

    with open(backtest_path) as f:
        backtest = json.load(f)

    for strat in backtest.get("strategies", []):
        if _strategy_key(strat["strategy"]) == strategy_key:
            bet_outcomes = _synthesize_bet_outcomes(strat, bet_amount=100)
            sim = simulate_bankroll(
                bet_outcomes,
                n_simulations=10000,
                initial_bankroll=bankroll,
                bet_size=bet_size,
            )
            sim["strategy_name"] = strat["strategy"]
            sim["strategy_key"] = strategy_key
            sim["source_hit_rate"] = strat["hit_rate_pct"]
            sim["source_roi"] = strat["roi_pct"]
            return sim

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = save_results()
    for key, sim in results.items():
        print(f"\n{sim['strategy_name']} ({key}):")
        print(f"  Mean final bankroll: ¥{sim['mean_final']:,.0f}")
        print(f"  Ruin probability:    {sim['ruin_probability']*100:.1f}%")
        print(f"  Profit probability:  {sim['profit_probability']*100:.1f}%")
        print(f"  Optimal bet size:    ¥{sim['optimal_bet_size']:,}")
        print(f"  5th-95th pctl:       ¥{sim['percentile_5']:,.0f} - ¥{sim['percentile_95']:,.0f}")
