# F1 Dashboard — CLAUDE.md

## Project Overview

A single-page Streamlit dashboard for interrogating historical F1 race data. All application logic lives in `f1app.py`. Data is sourced from the FastF1 library (cached locally) and the Ergast API.

## Running the App

```bash
source venv/bin/activate
streamlit run f1app.py
```

## Dependencies

Managed via `requirements.txt`. Key libraries:

- `fastf1` — F1 telemetry and session data
- `streamlit` — UI framework
- `plotly` — Interactive charts (track map, speed difference, standings heatmaps)
- `matplotlib` / `seaborn` — Static analysis plots (speed traces, tyre strategy, lap times)
- `pandas` / `numpy` — Data manipulation
- `timple` — Timedelta formatting via `strftimedelta`

Install: `pip install -r requirements.txt`

Note: `streamlit` is not in `requirements.txt` and must be installed separately: `pip install streamlit`

## Architecture

Everything is in `f1app.py`. There are no separate modules, pages, or config files. The file is structured as:

1. Imports and cache setup
2. Helper/utility functions
3. Data-fetching and plotting functions
4. Streamlit UI (sidebar controls + expander sections at the bottom of the file)

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
- **Year** (slider, 1985–2026)
- **Race** (selectbox from schedule, excludes testing events)
- **Session** (Race, FP1–3, Qualifying, Sprint, etc.)
- **Driver 1 / Driver 2** (abbreviations loaded from session results without full telemetry)

Main area uses `st.expander` sections:
1. Driver Standings — Ergast heatmap
2. Team Standings — Ergast heatmap with team colours
3. Track Map — Plotly, sector-coloured, uses FP1 data
4. Session Overview — Race position chart, notable events table, fastest laps table (clickable), driver lap times violin, tyre strategy
5. Lap Comparison — Single driver lap table + telemetry speed trace for selected lap(s)
6. Driver Comparison — Qualifying deltas, speed trace overlay, speed difference chart, sector times table
7. Championship — Who can still mathematically win

## Key Functions

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
