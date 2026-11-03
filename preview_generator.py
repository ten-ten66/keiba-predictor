"""
AI Race Preview Text Generator
================================
Generate human-readable race previews in Japanese using prediction data.

Usage:
    from preview_generator import generate_preview, generate_all_previews
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def generate_preview(race_info: Dict, predictions: List[Dict], pace_analysis: Optional[Dict] = None) -> str:
    """Generate a human-readable race preview in Japanese.

    Args:
        race_info: dict with keys like venue, race_name, course_type, distance, race_number
        predictions: list of dicts with horse_number, horse_name, composite_score, prob_win, prob_top3, odds, etc.
        pace_analysis: optional dict with predicted_pace, pace_advantage, horses info

    Returns:
        Formatted Japanese text preview string
    """
    top3 = sorted(predictions, key=lambda x: x.get('composite_score', x.get('score', 0)), reverse=True)[:3]

    if not top3:
        return ""

    venue = race_info.get('venue', '')
    race_name = race_info.get('race_name', '')
    race_num = race_info.get('race_number', '')
    course_type = race_info.get('course_type', '')
    distance = race_info.get('distance', '')

    header = f"【{venue} {race_name}】"
    if race_num:
        header = f"【{venue} {race_num}R {race_name}】"

    preview = header + "\n"
    preview += f"{course_type} {distance}m\n\n"

    # Main contender (honmei)
    h1 = top3[0]
    prob_win_pct = h1.get('prob_win', h1.get('win_prob', 0)) * 100
    preview += f"◎ 本命: {h1.get('horse_number', '?')}番 {h1.get('horse_name', '?')}\n"
    preview += f"  勝率{prob_win_pct:.1f}%"

    odds1 = h1.get('odds', 0)
    if odds1 > 0:
        preview += f" | オッズ{odds1:.1f}倍"

    if pace_analysis:
        horses_pace = pace_analysis.get('horses', {})
        h1_pace = horses_pace.get(str(h1.get('horse_number', '')), {})
        style = h1_pace.get('style', '')
        if style:
            preview += f" | 脚質: {style}"

    preview += "\n"

    # Counter pick (taikou)
    if len(top3) > 1:
        h2 = top3[1]
        prob2 = h2.get('prob_win', h2.get('win_prob', 0)) * 100
        preview += f"\n○ 対抗: {h2.get('horse_number', '?')}番 {h2.get('horse_name', '?')}"
        preview += f" (勝率{prob2:.1f}%"
        odds2 = h2.get('odds', 0)
        if odds2 > 0:
            preview += f" オッズ{odds2:.1f}倍"
        preview += ")\n"

    # Dark horse (tanana)
    if len(top3) > 2:
        h3 = top3[2]
        prob3 = h3.get('prob_win', h3.get('win_prob', 0)) * 100
        preview += f"▲ 単穴: {h3.get('horse_number', '?')}番 {h3.get('horse_name', '?')}"
        preview += f" (勝率{prob3:.1f}%"
        odds3 = h3.get('odds', 0)
        if odds3 > 0:
            preview += f" オッズ{odds3:.1f}倍"
        preview += ")\n"

    # Pace outlook
    if pace_analysis:
        predicted_pace = pace_analysis.get('predicted_pace', '')
        pace_advantage = pace_analysis.get('pace_advantage', '')
        if predicted_pace or pace_advantage:
            preview += f"\n【展開予想】{predicted_pace}"
            if pace_advantage:
                preview += f" → {pace_advantage}"
            preview += "\n"

    # Recommended bets
    preview += f"\n【推奨買い目】\n"
    preview += f"  単勝: {top3[0].get('horse_number', '?')}\n"

    if len(top3) >= 2:
        nums = sorted([top3[0].get('horse_number', 0), top3[1].get('horse_number', 0)])
        preview += f"  馬連: {nums[0]}-{nums[1]}\n"

    if len(top3) >= 3:
        nums3 = sorted([h.get('horse_number', 0) for h in top3])
        preview += f"  三連複: {nums3[0]}-{nums3[1]}-{nums3[2]}\n"

    # Confidence level
    avg_prob = sum(h.get('prob_win', h.get('win_prob', 0)) for h in top3) / len(top3)
    if avg_prob > 0.2:
        preview += "\n★ 信頼度: 高\n"
    elif avg_prob > 0.12:
        preview += "\n★ 信頼度: 中\n"
    else:
        preview += "\n★ 信頼度: 低\n"

    return preview


def generate_all_previews(races: List[Dict]) -> List[Dict]:
    """Generate previews for multiple races.

    Args:
        races: list of dicts, each with 'race_info', 'predictions', and optionally 'pace_analysis'

    Returns:
        list of dicts with 'race_id', 'race_info', 'preview_text'
    """
    results = []
    for race in races:
        race_info = race.get('race_info', {})
        predictions = race.get('predictions', [])
        pace = race.get('pace_analysis', None)
        text = generate_preview(race_info, predictions, pace)
        results.append({
            'race_id': race.get('race_id', ''),
            'race_info': race_info,
            'preview_text': text,
        })
    return results


def save_previews(previews: List[Dict], date_str: str, data_dir: str = "data"):
    """Save previews to data/previews/{date}.json.

    Args:
        previews: list from generate_all_previews
        date_str: YYYYMMDD format
        data_dir: base data directory
    """
    previews_dir = os.path.join(data_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)
    out_path = os.path.join(previews_dir, f"{date_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(previews, f, ensure_ascii=False, indent=2)
    return out_path


def load_previews(date_str: str, data_dir: str = "data") -> Optional[List[Dict]]:
    """Load cached previews for a date.

    Args:
        date_str: YYYYMMDD format
        data_dir: base data directory

    Returns:
        list of preview dicts, or None if not found
    """
    path = os.path.join(data_dir, "previews", f"{date_str}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
