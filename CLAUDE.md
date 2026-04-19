# F1 Dashboard — CLAUDE.md

## Project Overview

A Streamlit dashboard for interrogating historical F1 race data. Core application logic lives in `f1app.py`; ML-based race prediction logic is in `f1_predictor.py`. Data is sourced from the FastF1 library (cached locally), the Ergast API, and the Open-Meteo weather API.

## Running the App

```bash
source venv/bin/activate
streamlit run f1app.py
```

## Dependencies

Managed via `requirements.txt`. Key libraries:

- `fastf1` — F1 telemetry and session data
- `streamlit` — UI framework
- `plotly` — Interactive charts (track map, speed difference, standings heatmaps, podium visual)
- `matplotlib` / `seaborn` — Static analysis plots (speed traces, tyre strategy, lap times)
- `pandas` / `numpy` — Data manipulation
- `scikit-learn` — GradientBoostingClassifier for race prediction (`f1_predictor.py`)
- `requests` — Open-Meteo weather API calls (`f1_predictor.py`)
- `feedparser` — Parses Autosport F1 RSS feed for the news section
- `timple` — Timedelta formatting via `strftimedelta`

Install: `pip install -r requirements.txt`

Note: `streamlit` is not in `requirements.txt` and must be installed separately: `pip install streamlit`

## Architecture

Two Python files:

- **`f1app.py`** — all Streamlit UI and F1 data logic. Structured as:
  1. Imports and cache setup
  2. Helper/utility functions
  3. Data-fetching and plotting functions
  4. `_strip_html()`, `fetch_f1_news()`, `_show_f1_news()` — news feed helpers
  5. `_build_podium_figure()` and `_show_race_prediction()` (prediction UI helpers)
  6. Streamlit UI (sidebar controls + expander sections at the bottom of the file)

- **`f1_predictor.py`** — self-contained ML module for race podium prediction. Imported into `f1app.py` as `predictor`. Contains no Streamlit UI logic except `@st.cache_data` / `@st.cache_resource` decorators for caching.

### Caching

- `@st.cache_data` — lightweight API calls (`getschedule`, `get_event_driver_abbreviations`)
- `@st.cache_resource(max_entries=3)` — full session loads with telemetry (`load_session_cached`)

FastF1 disk cache is stored in `f1cache/` (committed as an empty dir via `.gitkeep`).

### Chart Libraries

Both Plotly and Matplotlib are used. Prefer Plotly for new interactive charts. Matplotlib is used for existing plots (speed traces, tyre strategy, violin plots, race position). Always call `plt.close(fig)` after `st.pyplot(fig)` to avoid memory leaks.

### Colours

- Driver/team colours: `fastf1.plotting.get_driver_color()`, `fpl.get_team_color()`
- Compound colours: `fastf1.plotting.get_compound_color()` / `fpl.get_compound_mapping()`
- Sector colours (consistent across the app): S1 `#4FC3F7`, S2 `#FFF176`, S3 `#CE93D8`
- Constructor hex colours are defined in `_CONSTRUCTOR_COLORS` dict for Ergast-based standings

## UI Structure

Sidebar controls drive everything:
- **Year** (slider, 2010–2026)
- **Race** (selectbox from schedule, excludes testing events)
- **Session** (Race, FP1–3, Qualifying, Sprint, etc.) — only shown for years ≥ 2018
- **Driver 1 / Driver 2** (abbreviations loaded from session results without full telemetry) — only shown for years ≥ 2018

### Year gating

For **years < 2018**, only Driver Standings and Team Standings are shown, plus an info message. All session/telemetry sections are hidden because FastF1 data is too sparse for reliable display.

Main area uses `st.expander` sections:
1. Driver Standings — Ergast heatmap (all years)
2. Team Standings — Ergast heatmap with team colours (all years)
3. Championship — Who can still mathematically win (all years)
4. F1 News (Autosport) — Latest articles from Autosport RSS feed, cached 15 min (all years)
5. Next Race Prediction — ML podium prediction with Plotly podium steps visual (all years)
6. Selection Details — Year, country, date, round, race name, and race weather summary (air temp, track temp, humidity, wind speed, rain indicator — loaded from Race session weather_data, years ≥ 2018 only)
7. Track Map — Plotly, sector-coloured, uses FP1 data (years ≥ 2018)
8. Session Overview — Race position chart, notable events table, fastest laps table (clickable), driver lap times violin, tyre strategy (years ≥ 2018)
9. Lap Comparison — Single driver lap table + telemetry speed trace for selected lap(s) (years ≥ 2018)
10. Driver Comparison — Qualifying deltas, speed trace overlay, speed difference chart, sector times table (years ≥ 2018)

## Key Functions

### `f1app.py`

| Function | Purpose |
|---|---|
| `load_session_cached(year, event, session_name)` | Loads full FastF1 session (telemetry included) |
| `drawtrackfor(session)` | Plotly track map with sector colouring and corner labels |
| `getSpeedTraceFor(session, d1, d2)` | Matplotlib speed trace comparison |
| `getSpeedDifferenceChart(session, d1, d2)` | Plotly speed delta chart with shaded fills |
| `showSectorTimesComparison(session, d1, d2)` | Styled DataFrame: fastest lap + theoretical best sectors |
| `showdriverstanding(year, round)` | Ergast driver championship heatmap |
| `showteamstanding(year, round)` | Ergast constructor championship heatmap |
| `tyreStrategies(session)` | Matplotlib horizontal bar tyre strategy chart |
| `driverlaptimes(session)` | Seaborn violin + swarm plot for top-10 finishers |
| `fastestlapstable(session)` | DataFrame of each driver's fastest lap (clickable — shows lap times chart) |
| `plot_driver_race_laps(session, driver)` | Plotly scatter of all quick laps for a driver, coloured by compound |
| `marshal_sector_location(sector_num, circuit_info)` | Maps a marshal sector number to a corner range e.g. "Between T3 & T4" |
| `calculatemaxpointsforremainingseason(year, round)` | Max points still available |
| `get_event_driver_abbreviations(year, event, session)` | Loads driver list without full telemetry |
| `_strip_html(text)` | Strips HTML tags and decodes entities from RSS feed summaries |
| `fetch_f1_news(max_items)` | Fetches and caches Autosport F1 RSS feed (ttl=900s); returns list of article dicts |
| `_show_f1_news()` | Renders the F1 News expander UI with headlines, dates, and summaries |
| `_build_podium_figure(predictions, driver_colors)` | Plotly podium steps visual (P2/P1/P3 blocks in team colours) |
| `_show_race_prediction()` | Renders the full Race Prediction expander UI |

### `f1_predictor.py`

| Function | Purpose |
|---|---|
| `fetch_historical_results(current_year, num_seasons)` | Pulls Ergast race results for up to 3 seasons; cached |
| `build_wet_labels(results_df)` | Labels each historical race wet/dry via FastF1 `weather_data` (no telemetry); cached |
| `engineer_features(df)` | Adds cumulative points (no leakage), rolling recent form, circuit avg finish, DNF rate, podium/win targets |
| `compute_sample_weights(df, current_year, reg_change_boost)` | Current season × boost (default 3×) + within-season `sqrt(round/max_round)`; prior seasons 0.3× / 0.1× |
| `train_podium_model(df, weights)` | GradientBoostingClassifier: predicts podium (top 3) probability |
| `train_win_model(df, weights)` | GradientBoostingClassifier: predicts win probability |
| `run_prediction_pipeline(current_year, num_seasons, reg_change_boost)` | Full cached pipeline: fetch → label → engineer → train. Returns `(podium_model, win_model, df, weights)` |
| `get_next_race_info(year)` | Next race from FastF1 schedule with lat/lon for weather lookup |
| `fetch_race_weather(lat, lon, race_date)` | Open-Meteo free API; returns forecast dict or `None` if >16 days away |
| `build_next_race_features(results_df, next_race, year, driver_grid_map, wet)` | Constructs feature rows for each driver for the next race |
| `predict_podium(podium_model, win_model, features_df)` | Returns DataFrame with `win_prob` and `podium_prob` per driver, sorted by win prob |
| `get_feature_importances(model)` | Returns feature name → importance DataFrame for the given model |

## Conventions

- Time formatting uses `strftimedelta(x, '%m:%s.%ms')` for lap times, `'%s.%ms'` for sector times
- `format_td_safe()` is a null-safe wrapper for timedelta formatting
- `rotate(xy, angle=...)` handles track coordinate rotation from circuit metadata
- Session data is always loaded via `load_session_cached` — never call `session.load()` directly in UI code
- Driver list for sidebar is loaded lightweight (no telemetry) via `get_event_driver_abbreviations`

## Known Gotchas

- **`round` is shadowed** — the global `round = int(race_info['RoundNumber'])` at the bottom of the file overwrites Python's built-in `round()`. Never use `round()` in any function — use `int()` instead.
- **Session Overview chart order** — maintained in `showracedetails()`: race position → notable events → fastest laps (clickable) → driver lap times violin → tyre strategy.
- **Notable events table** — filters `session.race_control_messages` for `SafetyCar` category and `RED`, `CHEQUERED`, `DOUBLE YELLOW` flags. Double yellow rows get a `Location` column derived from marshal sector distances cross-referenced with corner distances via `marshal_sector_location()`.
- **Clickable fastest laps table** — uses `st.dataframe(on_select="rerun", selection_mode="single-row")`. Selected row index from Streamlit is 0-based so always use `.iloc[]` not `.loc[]` (table index starts at 1).
- **Missing telemetry (older/some sessions)** — `get_car_data()`, `get_pos_data()`, and `session.get_circuit_info()` all raise `KeyError` when FastF1 has no data for that driver/session. `drawtrackfor`, `getSpeedTraceFor`, and `getSpeedDifferenceChart` all catch `KeyError` and return `None`; callers check for `None` and show `st.warning()` instead of crashing.
- **Pre-2019 tyre compounds** — older races use compound names like `SUPERSOFT`, `ULTRASOFT`, `HYPERSOFT` rather than `SOFT`/`MEDIUM`/`HARD`. `driverlaptimes()` builds `hue_order` dynamically from the actual compounds present and filters laps to only those in the compound mapping, to avoid seaborn's `NaN is not in list` error.
- **`run_prediction_pipeline` is `@st.cache_resource`** — it caches trained model objects across reruns. Changing `num_seasons` or `reg_change_boost` in the UI will retrain because they are part of the cache key. The wet label step (`build_wet_labels`) loads FastF1 sessions lightly (no telemetry) and is separately `@st.cache_data` so it survives app restarts.
- **Open-Meteo weather is only available within 16 days** — `fetch_race_weather` returns `None` beyond that window and the UI falls back to a manual wet/dry toggle.
- **`round` shadow applies in `f1_predictor.py` too** — that module never imports from `f1app.py` and doesn't shadow `round`, but be aware if you ever cross-import.
- **Streamlit API versions** — use `width='stretch'` not `use_container_width=True` for `st.plotly_chart` and `st.dataframe`. Use `st.html()` not `components.html()` for rendering raw HTML (e.g. styled DataFrames via `.to_html()`). `st.components.v1` is no longer imported.
