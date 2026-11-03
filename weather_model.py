"""
Micro-Weather Integration
=========================
競馬場近隣の気象データを取得し、特徴量として付与する。

データソース優先順位:
  1. JMA AMeDAS（無料・APIキー不要）
  2. Open-Meteo（無料・APIキー不要・予報データも取得可能）
  3. OpenWeatherMap（APIキーが必要）
  4. フォールバック（全APIが利用不可時は None 値のプレースホルダー）

Usage:
  # 気象データ取得
  from weather_model import get_weather_features_for_race, add_weather_features_to_df

  # 単一レース用
  feats = get_weather_features_for_race(venue="東京", course_type="芝")

  # DataFrame に一括付与
  df = add_weather_features_to_df(df, venue_col="venue", course_type_col="course_type")
"""

import os
import requests
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# OpenWeatherMap APIキー（環境変数 OWM_API_KEY から取得。未設定でもフォールバック動作）
OWM_API_KEY: Optional[str] = os.environ.get("OWM_API_KEY")

# AMeDAS station IDs near each racecourse
VENUE_STATIONS = {
    "東京": "44132",   # Fuchu
    "中山": "45106",   # Funabashi
    "阪神": "63518",   # Takarazuka
    "京都": "61286",   # Kyoto
    "中京": "51106",   # Nagoya
    "小倉": "82182",   # Kitakyushu
    "新潟": "54232",   # Niigata
    "福島": "36127",   # Fukushima
    "函館": "23232",   # Hakodate
    "札幌": "14163",   # Sapporo
}


def get_amedas_weather(venue, hours_before=3):
    """
    Get detailed weather data from nearest AMeDAS station.

    Parameters
    ----------
    venue : str
        競馬場名 (e.g. "東京", "中山")
    hours_before : int
        直近何時間分のデータを参照するか

    Returns
    -------
    dict or None
        気象データ辞書。取得失敗時は None。
    """
    station_id = VENUE_STATIONS.get(venue)
    if not station_id:
        logger.warning(f"Unknown venue for weather lookup: {venue}")
        return None

    now = datetime.now()
    # JMA AMeDAS latest JSON endpoint
    # The data URL uses yyyyMMdd_HH format and returns the latest 3-hour block
    date_str = now.strftime("%Y%m%d")
    hour_block = (now.hour // 3) * 3  # round down to 3-hour block
    url = (
        f"https://www.jma.go.jp/bosai/amedas/data/point/"
        f"{station_id}/{date_str}_{hour_block:02d}.json"
    )

    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "keiba-predictor/1.0"
        })
        if r.status_code != 200:
            logger.warning(f"AMeDAS HTTP {r.status_code} for {venue} ({station_id})")
            return None

        data = r.json()

        # AMeDAS JSON returns time-keyed entries
        # Each key is like "20260320120000" and value has temp, humidity, etc.
        if not data:
            return None

        # Get the most recent entry
        sorted_keys = sorted(data.keys(), reverse=True)
        latest_key = sorted_keys[0]
        latest = data[latest_key]

        # Collect rainfall over recent hours
        rainfall_values = []
        for key in sorted_keys[:hours_before]:
            entry = data[key]
            precip = entry.get("precipitation1h")
            if precip and len(precip) > 0 and precip[0] is not None:
                rainfall_values.append(precip[0])

        temp_val = latest.get("temp", [None])
        humidity_val = latest.get("humidity", [None])
        wind_val = latest.get("wind", [None, None])
        wind_dir_val = latest.get("windDirection", [None])
        precip_1h_val = latest.get("precipitation1h", [None])

        result = {
            "station_id": station_id,
            "venue": venue,
            "timestamp": latest_key,
            "temperature": temp_val[0] if temp_val else None,
            "humidity": humidity_val[0] if humidity_val else None,
            "rainfall_1h": precip_1h_val[0] if precip_1h_val else None,
            "rainfall_3h": sum(rainfall_values) if rainfall_values else 0.0,
            "wind_speed": wind_val[0] if wind_val else None,
            "wind_direction": wind_dir_val[0] if wind_dir_val else None,
        }
        logger.info(f"AMeDAS weather for {venue}: temp={result['temperature']}, "
                     f"rain_3h={result['rainfall_3h']}, wind={result['wind_speed']}")
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"AMeDAS timeout for {venue}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"AMeDAS connection error for {venue}")
        return None
    except Exception as e:
        logger.error(f"AMeDAS error for {venue}: {e}")
        return None


def assess_track_impact(weather, course_type="芝"):
    """
    Assess how weather affects track conditions.

    Parameters
    ----------
    weather : dict
        get_amedas_weather() の戻り値
    course_type : str
        "芝" or "ダート"

    Returns
    -------
    dict
        馬場影響評価
    """
    if weather is None:
        return {
            "moisture_level": "unknown",
            "front_runner_impact": 0,
            "closer_impact": 0,
            "wind_effect": "unknown",
            "summary": "気象データ取得不可",
        }

    impact = {
        "moisture_level": "dry",
        "front_runner_impact": 0,   # negative = disadvantage
        "closer_impact": 0,
        "wind_effect": "calm",
        "summary": "",
    }

    rainfall = weather.get("rainfall_3h", 0) or 0
    wind = weather.get("wind_speed", 0) or 0
    temp = weather.get("temperature")
    humidity = weather.get("humidity")

    # --- Rainfall impact ---
    if rainfall > 20:
        impact["moisture_level"] = "very_heavy"
        if course_type == "ダート":
            impact["front_runner_impact"] = -20
            impact["closer_impact"] = 15
        else:  # 芝
            impact["front_runner_impact"] = -15
            impact["closer_impact"] = 10
    elif rainfall > 10:
        impact["moisture_level"] = "heavy"
        if course_type == "ダート":
            impact["front_runner_impact"] = -15
            impact["closer_impact"] = 10
        else:
            impact["front_runner_impact"] = -10
            impact["closer_impact"] = 8
    elif rainfall > 3:
        impact["moisture_level"] = "wet"
        impact["front_runner_impact"] = -8
        impact["closer_impact"] = 5
    elif rainfall > 0.5:
        impact["moisture_level"] = "slightly_wet"
        impact["front_runner_impact"] = -3
        impact["closer_impact"] = 2

    # --- Wind impact ---
    if wind > 8:
        impact["wind_effect"] = "very_strong"
        impact["front_runner_impact"] -= 8
    elif wind > 5:
        impact["wind_effect"] = "strong_headwind_possible"
        impact["front_runner_impact"] -= 5
    elif wind > 3:
        impact["wind_effect"] = "moderate"
        impact["front_runner_impact"] -= 2

    # --- Temperature effect (extreme heat or cold) ---
    if temp is not None:
        if temp > 33:
            impact["heat_stress"] = True
            impact["summary"] += "猛暑注意。スタミナ消耗が激しい。"
        elif temp < 2:
            impact["cold_stress"] = True
            impact["summary"] += "低温注意。馬場が硬い可能性。"

    # --- Build summary ---
    moisture_labels = {
        "dry": "良馬場（乾燥）",
        "slightly_wet": "やや湿り気味",
        "wet": "湿った馬場",
        "heavy": "重馬場",
        "very_heavy": "不良馬場相当",
    }
    impact["summary"] = moisture_labels.get(impact["moisture_level"], "") + "。"

    if impact["wind_effect"] in ("strong_headwind_possible", "very_strong"):
        impact["summary"] += f"強風（{wind}m/s）で逃げ馬不利。"

    if impact["front_runner_impact"] < -10:
        impact["summary"] += "差し・追込有利の馬場。"
    elif impact["front_runner_impact"] < -5:
        impact["summary"] += "やや差し有利。"

    return impact


def get_weather_and_impact(venue, course_type="芝", hours_before=3):
    """
    Convenience function: get weather + track impact in one call.

    Returns
    -------
    dict
        {"weather": {...}, "track_impact": {...}}
    """
    weather = get_amedas_weather(venue, hours_before=hours_before)
    track_impact = assess_track_impact(weather, course_type)
    return {
        "venue": venue,
        "course_type": course_type,
        "weather": weather,
        "track_impact": track_impact,
    }


# Fallback: return synthetic weather when API is unavailable
def get_fallback_weather(venue):
    """Return a default/unknown weather dict when AMeDAS is unreachable."""
    return {
        "station_id": VENUE_STATIONS.get(venue),
        "venue": venue,
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "temperature": None,
        "humidity": None,
        "rainfall_1h": None,
        "rainfall_3h": 0.0,
        "wind_speed": None,
        "wind_direction": None,
        "source": "fallback",
    }


# ---------------------------------------------------------------------------
# Open-Meteo（フリー・APIキー不要）
# ---------------------------------------------------------------------------

# 競馬場の緯度・経度（Open-Meteo / OpenWeatherMap 用）
VENUE_COORDS: Dict[str, Dict[str, float]] = {
    "東京": {"lat": 35.7394, "lon": 139.4884},
    "中山": {"lat": 35.7743, "lon": 140.0146},
    "阪神": {"lat": 34.7820, "lon": 135.3534},
    "京都": {"lat": 34.9049, "lon": 135.7226},
    "中京": {"lat": 35.1143, "lon": 136.8697},
    "小倉": {"lat": 33.8506, "lon": 130.7403},
    "新潟": {"lat": 37.8456, "lon": 138.9506},
    "福島": {"lat": 37.7500, "lon": 140.4667},
    "函館": {"lat": 41.7310, "lon": 140.7287},
    "札幌": {"lat": 43.0496, "lon": 141.3735},
}


def get_open_meteo_weather(venue: str, hours_ahead: int = 0) -> Optional[Dict[str, Any]]:
    """Open-Meteo から気象データを取得する（APIキー不要）。

    Parameters
    ----------
    venue : str
        競馬場名
    hours_ahead : int
        現在時刻から何時間後の予報を取得するか（0=現在）

    Returns
    -------
    dict or None
        気象データ辞書。取得失敗時は None。
    """
    coords = VENUE_COORDS.get(venue)
    if not coords:
        logger.warning(f"Open-Meteo: 未知の競馬場 {venue}")
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,precipitation_probability",
        "current_weather": "true",
        "timezone": "Asia/Tokyo",
        "forecast_days": 1,
    }

    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "keiba-predictor/1.0"})
        if r.status_code != 200:
            logger.warning(f"Open-Meteo HTTP {r.status_code} for {venue}")
            return None

        data = r.json()

        # 現在時刻に最も近い hourly インデックスを特定
        now = datetime.now()
        target_hour = now.replace(minute=0, second=0, microsecond=0)
        if hours_ahead > 0:
            target_hour = target_hour + timedelta(hours=hours_ahead)

        hourly_times = data.get("hourly", {}).get("time", [])
        target_str = target_hour.strftime("%Y-%m-%dT%H:00")
        idx = next((i for i, t in enumerate(hourly_times) if t == target_str), 0)

        def _safe(key):
            vals = data.get("hourly", {}).get(key, [])
            return vals[idx] if idx < len(vals) else None

        current = data.get("current_weather", {})

        result: Dict[str, Any] = {
            "venue": venue,
            "timestamp": target_str,
            "temperature": _safe("temperature_2m") or current.get("temperature"),
            "humidity": _safe("relativehumidity_2m"),
            "rainfall_1h": _safe("precipitation"),
            "rainfall_3h": None,  # hourly のみ提供のため合算は別途
            "wind_speed": _safe("windspeed_10m") or current.get("windspeed"),
            "wind_direction": current.get("winddirection"),
            "precip_prob": _safe("precipitation_probability"),
            "source": "open_meteo",
        }

        # 直近3時間降水量を合算
        precip_vals = [
            data.get("hourly", {}).get("precipitation", [])[max(0, idx - k)]
            for k in range(3)
            if idx - k >= 0 and idx - k < len(data.get("hourly", {}).get("precipitation", []))
        ]
        result["rainfall_3h"] = sum(v for v in precip_vals if v is not None)

        logger.info(
            f"Open-Meteo weather for {venue}: "
            f"temp={result['temperature']}, rain_1h={result['rainfall_1h']}, "
            f"precip_prob={result['precip_prob']}"
        )
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"Open-Meteo timeout for {venue}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"Open-Meteo connection error for {venue}")
        return None
    except Exception as e:
        logger.error(f"Open-Meteo error for {venue}: {e}")
        return None


def get_owm_weather(venue: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """OpenWeatherMap から気象データを取得する（APIキーが必要）。

    Parameters
    ----------
    venue : str
        競馬場名
    api_key : str or None
        APIキー。None の場合は環境変数 OWM_API_KEY を使用。

    Returns
    -------
    dict or None
        気象データ辞書。取得失敗時は None。
    """
    key = api_key or OWM_API_KEY
    if not key:
        logger.debug("OWM_API_KEY 未設定、OpenWeatherMap スキップ")
        return None

    coords = VENUE_COORDS.get(venue)
    if not coords:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": key,
        "units": "metric",
        "lang": "ja",
    }

    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "keiba-predictor/1.0"})
        if r.status_code == 401:
            logger.warning("OpenWeatherMap: APIキーが無効")
            return None
        if r.status_code != 200:
            logger.warning(f"OpenWeatherMap HTTP {r.status_code} for {venue}")
            return None

        data = r.json()
        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})

        result: Dict[str, Any] = {
            "venue": venue,
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
            "temperature": main.get("temp"),
            "humidity": main.get("humidity"),
            "rainfall_1h": rain.get("1h", 0.0),
            "rainfall_3h": rain.get("3h", 0.0),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "precip_prob": None,  # current weather API では提供されない
            "source": "openweathermap",
        }
        logger.info(
            f"OWM weather for {venue}: "
            f"temp={result['temperature']}, rain_1h={result['rainfall_1h']}"
        )
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"OpenWeatherMap timeout for {venue}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"OpenWeatherMap connection error for {venue}")
        return None
    except Exception as e:
        logger.error(f"OpenWeatherMap error for {venue}: {e}")
        return None


def get_best_weather(venue: str, course_type: str = "芝",
                     hours_ahead: int = 0) -> Dict[str, Any]:
    """利用可能な全APIを試し、最初に成功したデータを返す。

    取得順序:
      1. JMA AMeDAS（フリー・APIキー不要）
      2. Open-Meteo  （フリー・APIキー不要）
      3. OpenWeatherMap（OWM_API_KEY がある場合のみ）
      4. フォールバック（全て失敗時）

    Parameters
    ----------
    venue : str
        競馬場名
    course_type : str
        "芝" or "ダート"
    hours_ahead : int
        0=現在、正値=予報

    Returns
    -------
    dict
        気象データ辞書。必ず値を返す（失敗時はフォールバック）。
    """
    # 1. AMeDAS
    weather = get_amedas_weather(venue)
    if weather:
        weather["source"] = weather.get("source", "amedas")
        return weather

    # 2. Open-Meteo
    weather = get_open_meteo_weather(venue, hours_ahead=hours_ahead)
    if weather:
        return weather

    # 3. OpenWeatherMap（APIキーがある場合のみ）
    if OWM_API_KEY:
        weather = get_owm_weather(venue)
        if weather:
            return weather

    # 4. フォールバック
    logger.warning(f"全気象APIが利用不可: {venue} → フォールバック使用")
    return get_fallback_weather(venue)


def weather_to_features(weather: Dict[str, Any]) -> Dict[str, float]:
    """気象データ辞書を数値特徴量辞書に変換する。

    Parameters
    ----------
    weather : dict
        get_best_weather() 等の戻り値

    Returns
    -------
    dict
        モデル入力用の特徴量辞書。欠損値は 0.0（NaN補完済み）。

    生成カラム（data_processor.py の WEATHER_FEATURES と対応）:
      weather_temperature  : 気温 [°C]
      weather_humidity     : 湿度 [%]
      weather_rainfall_1h  : 1時間降水量 [mm]
      weather_rainfall_3h  : 3時間降水量 [mm]
      weather_wind_speed   : 風速 [m/s]
      weather_precip_prob  : 降水確率 [0-100]
      weather_available    : 1=取得成功, 0=フォールバック
    """
    if weather is None:
        weather = {}

    is_available = 1 if weather.get("source", "fallback") != "fallback" else 0

    def _float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return {
        "weather_temperature": _float(weather.get("temperature")),
        "weather_humidity":    _float(weather.get("humidity")),
        "weather_rainfall_1h": _float(weather.get("rainfall_1h")),
        "weather_rainfall_3h": _float(weather.get("rainfall_3h")),
        "weather_wind_speed":  _float(weather.get("wind_speed")),
        "weather_precip_prob": _float(weather.get("precip_prob")),
        "weather_available":   float(is_available),
    }


def get_weather_features_for_race(
    venue: str,
    course_type: str = "芝",
    hours_ahead: int = 0,
) -> Dict[str, float]:
    """競馬場と馬場タイプを指定して、モデル用気象特徴量を取得する。

    Parameters
    ----------
    venue : str
        競馬場名（"東京", "中山" 等）
    course_type : str
        "芝" or "ダート"
    hours_ahead : int
        0=現在、正値=レース開始時刻までの予報時間

    Returns
    -------
    dict
        {weather_temperature, weather_humidity, ...} 形式の特徴量辞書。
        APIが全て失敗してもフォールバック値（0.0 / available=0）を返す。
    """
    weather = get_best_weather(venue, course_type=course_type, hours_ahead=hours_ahead)
    return weather_to_features(weather)


def add_weather_features_to_df(
    df: "pd.DataFrame",
    venue_col: str = "venue",
    course_type_col: str = "course_type",
    hours_ahead: int = 0,
) -> "pd.DataFrame":
    """DataFrame の各行に気象特徴量カラムを付与する。

    同一 venue の行には同じ気象データを使い回す（APIコール数を最小化）。

    Parameters
    ----------
    df : pd.DataFrame
        venue_col と course_type_col を持つ DataFrame
    venue_col : str
        競馬場カラム名（デフォルト "venue"）
    course_type_col : str
        コースタイプカラム名（デフォルト "course_type"）
    hours_ahead : int
        0=現在、正値=予報

    Returns
    -------
    pd.DataFrame
        weather_* カラムが追加された DataFrame
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logger.error("pandas が必要です")
        return df

    df = df.copy()

    # 気象特徴量の初期化（フォールバック値で埋める）
    feat_cols = [
        "weather_temperature", "weather_humidity", "weather_rainfall_1h",
        "weather_rainfall_3h", "weather_wind_speed", "weather_precip_prob",
        "weather_available",
    ]
    for col in feat_cols:
        df[col] = 0.0

    if venue_col not in df.columns:
        logger.warning(f"カラム '{venue_col}' が見つからない、気象特徴量スキップ")
        return df

    course_col_exists = course_type_col in df.columns

    # venue ごとにAPIを1回だけ呼ぶ
    weather_cache: Dict[str, Dict[str, float]] = {}
    unique_venues = df[venue_col].dropna().unique()

    for venue in unique_venues:
        if not isinstance(venue, str) or not venue:
            continue
        ct = "芝"
        if course_col_exists:
            ct_vals = df.loc[df[venue_col] == venue, course_type_col].dropna()
            if not ct_vals.empty:
                ct = str(ct_vals.iloc[0])
        weather_cache[venue] = get_weather_features_for_race(
            venue, course_type=ct, hours_ahead=hours_ahead
        )
        logger.info(f"気象特徴量取得: {venue} → {weather_cache[venue]}")

    # DataFrame に書き込み
    for venue, feats in weather_cache.items():
        mask = df[venue_col] == venue
        for col, val in feats.items():
            df.loc[mask, col] = val

    n_with_weather = df["weather_available"].sum()
    logger.info(
        f"気象特徴量付与完了: {n_with_weather:.0f}/{len(df)} 行 "
        f"({n_with_weather / len(df) * 100:.1f}%)"
    )
    return df
