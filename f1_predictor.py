"""
f1_predictor.py
Race podium prediction module for the F1 Dashboard.

Uses Ergast historical race results + FastF1 weather data to train a
GradientBoostingClassifier. Predictions are made for the next upcoming race.

Weighting scheme:
  - Current season receives a boosted weight (reg-change multiplier)
  - Within the current season, later rounds are weighted more heavily
  - Prior seasons receive exponential decay
"""

import datetime
import warnings
import requests
import numpy as np
import pandas as pd
import fastf1 as f1
import streamlit as st
from fastf1.ergast import Ergast
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENT_YEAR = datetime.date.today().year

# Season weights: keyed by offset from current year (0 = current, -1 = last, -2 = two ago)
# These are base weights before the reg-change multiplier is applied.
_BASE_SEASON_WEIGHTS = {0: 1.0, -1: 0.3, -2: 0.1}

# ---------------------------------------------------------------------------
# Data fetching — Ergast
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_results(current_year: int, num_seasons: int = 3) -> pd.DataFrame:
    """
    Pull race results for the last `num_seasons` seasons from Ergast.
    Returns a flat DataFrame with one row per driver per race.
    Columns: year, round, circuitId, driverCode, constructorId,
             grid, position, points, status
    """
    ergast = Ergast()
    rows = []

    for offset in range(num_seasons - 1, -1, -1):
        year = current_year - offset
        try:
            schedule = ergast.get_race_schedule(year)
        except Exception:
            continue

        for rnd, _ in schedule['raceName'].items():
            round_num = rnd + 1
            try:
                res = ergast.get_race_results(season=year, round=round_num)
                if not res.content:
                    continue
                df = res.content[0].copy()
            except Exception:
                continue

            circuit_id = res.description.get('circuitId', [None])
            circuit_id = circuit_id[0] if len(circuit_id) > 0 else None

            for _, row in df.iterrows():
                pos = row.get('position', None)
                try:
                    pos = int(pos)
                except (TypeError, ValueError):
                    pos = 20  # DNF / DSQ treated as last

                status = str(row.get('status', '')).lower()
                dnf = 1 if ('retired' in status or 'accident' in status
                            or 'collision' in status or 'mechanical' in status
                            or 'engine' in status or 'gearbox' in status) else 0

                rows.append({
                    'year': year,
                    'round': round_num,
                    'circuitId': circuit_id,
                    'driverCode': row.get('driverCode', row.get('driverId', '')),
                    'constructorId': row.get('constructorId', ''),
                    'grid': int(row.get('grid', 10)),
                    'position': pos,
                    'points': float(row.get('points', 0)),
                    'dnf': dnf,
                })

    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def label_wet_race(year: int, round_num: int) -> bool:
    """
    Returns True if rain fell during the race session.
    Uses FastF1 weather_data (lightweight — no telemetry loaded).
    Falls back to False on any error.
    """
    try:
        session = f1.get_session(year, round_num, 'R')
        session.load(weather=True, laps=False, telemetry=False, messages=False)
        wd = session.weather_data
        if wd is not None and not wd.empty and 'Rainfall' in wd.columns:
            return bool(wd['Rainfall'].any())
    except Exception:
        pass
    return False


@st.cache_data(ttl=3600, show_spinner=False)
def build_wet_labels(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'wet' boolean column to results_df by querying FastF1 per race.
    Only fetches for races with year >= 2018 (FastF1 data availability).
    """
    if results_df.empty:
        return results_df

    race_keys = results_df[['year', 'round']].drop_duplicates()
    wet_map = {}
    for _, r in race_keys.iterrows():
        y, rnd = int(r['year']), int(r['round'])
        if y >= 2018:
            wet_map[(y, rnd)] = label_wet_race(y, rnd)
        else:
            wet_map[(y, rnd)] = False

    results_df = results_df.copy()
    results_df['wet'] = results_df.apply(
        lambda r: wet_map.get((int(r['year']), int(r['round'])), False), axis=1
    ).astype(int)
    return results_df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_cumulative_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute the driver and constructor points accumulated
    *before* that race in that season (i.e. no data leakage).
    """
    df = df.sort_values(['year', 'round']).copy()

    for col_name, group_col in [('driver_pts_before', 'driverCode'),
                                 ('constructor_pts_before', 'constructorId')]:
        cumsum = (df.groupby(['year', group_col])['points']
                    .cumsum()
                    .shift(1)
                    .fillna(0))
        df[col_name] = cumsum

    return df


def _compute_recent_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Rolling mean of finishing position over the previous `n` races per driver.
    Lower is better (P1 = best).
    """
    df = df.sort_values(['year', 'round']).copy()
    df['recent_form'] = (
        df.groupby('driverCode')['position']
          .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    df['recent_form'] = df['recent_form'].fillna(10.0)
    return df


def _compute_circuit_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each driver's historical average finishing position at this circuit
    across all prior appearances (no leakage — shifted by 1 race).
    """
    df = df.sort_values(['year', 'round']).copy()

    def _circuit_rolling(group):
        return group['position'].shift(1).expanding().mean()

    df['circuit_avg_pos'] = (
        df.groupby(['driverCode', 'circuitId'], group_keys=False)
          .apply(_circuit_rolling)
    )
    df['circuit_avg_pos'] = df['circuit_avg_pos'].fillna(10.0)
    return df


def _compute_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling DNF rate per driver over prior 10 races."""
    df = df.sort_values(['year', 'round']).copy()
    df['dnf_rate'] = (
        df.groupby('driverCode')['dnf']
          .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df['dnf_rate'] = df['dnf_rate'].fillna(0.1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return a clean feature DataFrame."""
    df = _compute_cumulative_standings(df)
    df = _compute_recent_form(df)
    df = _compute_circuit_avg(df)
    df = _compute_dnf_rate(df)

    # Podium target: 1 if finished P1/P2/P3
    df['on_podium'] = (df['position'] <= 3).astype(int)
    # Win target
    df['is_win'] = (df['position'] == 1).astype(int)

    return df


FEATURE_COLS = [
    'grid',
    'driver_pts_before',
    'constructor_pts_before',
    'recent_form',
    'circuit_avg_pos',
    'dnf_rate',
    'wet',
]

FEATURE_LABELS = {
    'grid': 'Grid Position',
    'driver_pts_before': 'Driver Points (before race)',
    'constructor_pts_before': 'Constructor Points (before race)',
    'recent_form': 'Recent Form (avg finish, 5 races)',
    'circuit_avg_pos': 'Circuit Historical Avg Finish',
    'dnf_rate': 'DNF Rate (last 10 races)',
    'wet': 'Wet Race',
}


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(
    df: pd.DataFrame,
    current_year: int,
    reg_change_boost: float = 3.0,
) -> np.ndarray:
    """
    Compute a sample weight for each row.

    Formula:
      weight = season_base_weight * round_recency_factor

    season_base_weight:
      - current year  → 1.0 * reg_change_boost
      - current - 1   → 0.3
      - current - 2   → 0.1

    round_recency_factor (within current season only):
      sqrt(round / max_round)  — earlier rounds slightly down-weighted.
      Prior seasons: 1.0 (uniform within season).
    """
    weights = np.ones(len(df))
    max_round_current = df[df['year'] == current_year]['round'].max()
    if pd.isna(max_round_current):
        max_round_current = 1

    for i, row in df.iterrows():
        offset = row['year'] - current_year  # 0, -1, or -2
        base = _BASE_SEASON_WEIGHTS.get(offset, 0.05)

        if offset == 0:
            # Apply regulation-change boost + within-season recency
            recency = np.sqrt(row['round'] / max(max_round_current, 1))
            w = base * reg_change_boost * recency
        else:
            w = base

        weights[df.index.get_loc(i)] = w

    return weights


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_podium_model(
    df: pd.DataFrame,
    sample_weights: np.ndarray,
) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier to predict podium probability."""
    clean = df[FEATURE_COLS + ['on_podium']].dropna()
    idx = clean.index
    X = clean[FEATURE_COLS].values
    y = clean['on_podium'].values
    w = sample_weights[df.index.get_indexer(idx)]

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y, sample_weight=w)
    return model


def train_win_model(
    df: pd.DataFrame,
    sample_weights: np.ndarray,
) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier to predict win probability."""
    clean = df[FEATURE_COLS + ['is_win']].dropna()
    idx = clean.index
    X = clean[FEATURE_COLS].values
    y = clean['is_win'].values
    w = sample_weights[df.index.get_indexer(idx)]

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y, sample_weight=w)
    return model


# ---------------------------------------------------------------------------
# Next race info
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_next_race_info(year: int) -> dict | None:
    """
    Returns a dict describing the next race that hasn't happened yet,
    or None if the season is complete.
    Keys: round, name, circuitId, date, country, lat, lon
    """
    try:
        schedule = f1.get_event_schedule(year, include_testing=False)
    except Exception:
        return None

    today = datetime.date.today()
    future = schedule[pd.to_datetime(schedule['EventDate']).dt.date >= today]
    if future.empty:
        return None

    nxt = future.iloc[0]
    round_num = int(nxt['RoundNumber'])

    # Try to get circuit coordinates from FastF1
    lat, lon = None, None
    try:
        session = f1.get_session(year, round_num, 'R')
        # circuit_info gives us corner data but not lat/lon directly;
        # use the ergast circuit info instead
        ergast = Ergast()
        ci = ergast.get_race_schedule(year)
        # Ergast schedule has 'lat' and 'long' columns for circuit location
        row = ci[ci['round'] == round_num] if 'round' in ci.columns else pd.DataFrame()
        if not row.empty:
            lat = float(row.iloc[0].get('lat', 0) or 0)
            lon = float(row.iloc[0].get('long', 0) or 0)
    except Exception:
        pass

    return {
        'round': round_num,
        'name': nxt.get('EventName', ''),
        'circuitId': nxt.get('OfficialEventName', nxt.get('EventName', '')),
        'date': pd.to_datetime(nxt['EventDate']).date(),
        'country': nxt.get('Country', ''),
        'location': nxt.get('Location', ''),
        'lat': lat,
        'lon': lon,
    }


# ---------------------------------------------------------------------------
# Weather forecast
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_race_weather(lat: float, lon: float, race_date: datetime.date) -> dict | None:
    """
    Fetch weather forecast from Open-Meteo for the given coordinates and date.
    Returns dict with keys: temp_c, precipitation_mm, wind_kph, is_wet, description
    Returns None if the date is too far ahead (>16 days) or on error.
    """
    if lat is None or lon is None:
        return None

    days_ahead = (race_date - datetime.date.today()).days
    if days_ahead > 16:
        return None  # Beyond reliable forecast window

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max"
        f"&timezone=auto"
        f"&forecast_days={min(days_ahead + 1, 16)}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get('daily', {})
        dates = daily.get('time', [])
        date_str = race_date.isoformat()
        if date_str not in dates:
            return None
        idx = dates.index(date_str)
        temp = daily['temperature_2m_max'][idx]
        precip = daily['precipitation_sum'][idx]
        wind = daily['wind_speed_10m_max'][idx]
        is_wet = precip > 0.5  # >0.5mm = wet conditions
        if is_wet:
            desc = f"Rain expected ({precip:.1f}mm)"
        else:
            desc = f"Dry ({temp:.0f}°C, wind {wind:.0f} km/h)"
        return {
            'temp_c': temp,
            'precipitation_mm': precip,
            'wind_kph': wind,
            'is_wet': is_wet,
            'description': desc,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build prediction input for next race
# ---------------------------------------------------------------------------

def build_next_race_features(
    results_df: pd.DataFrame,
    next_race: dict,
    current_year: int,
    driver_grid_map: dict,
    wet: bool,
) -> pd.DataFrame:
    """
    Build a feature row for each driver in the next race.

    `driver_grid_map`: {driverCode: expected_grid_position}
    Uses the most recent available stats for each driver from results_df.
    """
    # Current season results only (for points)
    current = results_df[results_df['year'] == current_year].copy()

    # Latest points per driver/constructor in current season
    if not current.empty:
        latest_round = current['round'].max()
        before_race = current[current['round'] <= latest_round]
        driver_pts = (before_race.groupby('driverCode')['points']
                                 .sum().to_dict())
        constructor_pts = (before_race.groupby('constructorId')['points']
                                      .sum().to_dict())
        # Map driver -> constructorId from most recent appearance
        driver_constructor = (
            before_race.sort_values('round')
                       .groupby('driverCode')['constructorId']
                       .last().to_dict()
        )
    else:
        driver_pts = {}
        constructor_pts = {}
        driver_constructor = {}

    # Recent form (last 5 races overall)
    recent_form_map = {}
    for drv, grp in results_df.sort_values(['year', 'round']).groupby('driverCode'):
        last5 = grp['position'].tail(5).mean()
        recent_form_map[drv] = last5 if not pd.isna(last5) else 10.0

    # Circuit historical average
    circuit_id = next_race.get('circuitId', '')
    circuit_hist = results_df[results_df['circuitId'] == circuit_id]
    circuit_avg_map = {}
    if not circuit_hist.empty:
        circuit_avg_map = (circuit_hist.groupby('driverCode')['position']
                                        .mean().to_dict())

    # DNF rate (last 10 races per driver)
    dnf_rate_map = {}
    for drv, grp in results_df.sort_values(['year', 'round']).groupby('driverCode'):
        last10 = grp['dnf'].tail(10).mean()
        dnf_rate_map[drv] = last10 if not pd.isna(last10) else 0.1

    rows = []
    for drv, grid in driver_grid_map.items():
        constructor = driver_constructor.get(drv, '')
        rows.append({
            'driverCode': drv,
            'constructorId': constructor,
            'grid': grid,
            'driver_pts_before': driver_pts.get(drv, 0),
            'constructor_pts_before': constructor_pts.get(constructor, 0),
            'recent_form': recent_form_map.get(drv, 10.0),
            'circuit_avg_pos': circuit_avg_map.get(drv, 10.0),
            'dnf_rate': dnf_rate_map.get(drv, 0.1),
            'wet': int(wet),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_podium(
    podium_model: GradientBoostingClassifier,
    win_model: GradientBoostingClassifier,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      driverCode, constructorId, win_prob, podium_prob
    Sorted by win_prob descending.
    """
    X = features_df[FEATURE_COLS].values
    win_probs = win_model.predict_proba(X)[:, 1]
    podium_probs = podium_model.predict_proba(X)[:, 1]

    out = features_df[['driverCode', 'constructorId']].copy()
    out['win_prob'] = win_probs
    out['podium_prob'] = podium_probs
    return out.sort_values('win_prob', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importances(model: GradientBoostingClassifier) -> pd.DataFrame:
    """Returns a DataFrame of feature name → importance, sorted descending."""
    imp = model.feature_importances_
    return (pd.DataFrame({'feature': [FEATURE_LABELS[c] for c in FEATURE_COLS],
                          'importance': imp})
              .sort_values('importance', ascending=False)
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Full pipeline (cached end-to-end result)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def run_prediction_pipeline(
    current_year: int,
    num_seasons: int,
    reg_change_boost: float,
) -> tuple:
    """
    Fetch data, engineer features, train models.
    Returns (podium_model, win_model, engineered_df, sample_weights).
    Cached as a resource so training doesn't re-run on every rerun.
    """
    results = fetch_historical_results(current_year, num_seasons)
    if results.empty:
        return None, None, None, None

    results = build_wet_labels(results)
    results = engineer_features(results)
    weights = compute_sample_weights(results, current_year, reg_change_boost)

    podium_model = train_podium_model(results, weights)
    win_model = train_win_model(results, weights)

    return podium_model, win_model, results, weights
