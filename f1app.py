import os
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1 as f1
import fastf1.plotting as fpl
from fastf1.core import Laps
from fastf1.ergast import Ergast
from timple.timedelta import strftimedelta
from typing import List
from plotly.io import show
from matplotlib.colors import to_rgb
import f1_predictor as predictor

os.makedirs('f1cache', exist_ok=True)
f1.Cache.enable_cache('f1cache')

def fastest_and_mins(drv: pd.DataFrame):
    """Return fastest lap row (Series) and min sector times as dict."""
    if drv.empty:
        return None, {'Sector1Time': pd.NaT, 'Sector2Time': pd.NaT, 'Sector3Time': pd.NaT, 'LapTime': pd.NaT}
    fastest_row = drv.sort_values('LapTime', ascending=True).iloc[0]
    mins = drv[['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']].min()
    return fastest_row, mins.to_dict()

def format_td_safe(x, fmt_seconds='%s.%ms', fmt_lap='%m:%s.%ms'):
    if pd.isna(x):
        return ''
    # x may already be a string (formatted) or timedelta or float seconds
    if isinstance(x, str):
        return x
    try:
        # assume timedelta-like
        return strftimedelta(x, fmt_seconds)
    except Exception:
        try:
            return str(x)
        except Exception:
            return ''

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def getsessiondata(year, event, session, verbose=False):
    if verbose:
        print("Getting data for", event, year)
        event = f1.get_event(year, event)
        print(event)
        return False
    session = f1.get_session(year, event, session)
    return session

@st.cache_data
def getschedule(year):
    return f1.get_event_schedule(year)


@st.cache_resource(max_entries=2)
def load_session_cached(year, event, session_name):
    session = f1.get_session(year, event, session_name)
    session.load()
    return session

def calendardetails(year, verbose=False):
    calendar = f1.get_event_schedule(year)
    if verbose:
        print(calendar)
    return calendar

def getlapsfor(session, driver):
    laps = session.laps.pick_driver(driver)
    return laps

def drawtrackfor(session):
    lap = session.laps.pick_fastest()
    try:
        pos = lap.get_pos_data()
    except KeyError:
        return None
    circuit_info = session.get_circuit_info()

    track_angle = circuit_info.rotation / 180 * np.pi

    # Sector boundary times
    s1_end = lap['Sector1Time']
    s2_end = lap['Sector1Time'] + lap['Sector2Time']

    sector_colors = ['#4FC3F7', '#FFF176', '#CE93D8']  # S1 light blue, S2 yellow, S3 purple
    sector_names = ['Sector 1', 'Sector 2', 'Sector 3']

    masks = [
        pos['Time'] <= s1_end,
        (pos['Time'] > s1_end) & (pos['Time'] <= s2_end),
        pos['Time'] > s2_end,
    ]

    # Build Plotly figure to avoid matplotlib/timple unit conversion recursion in some environments.
    fig = go.Figure()

    for i, (mask, color, name) in enumerate(zip(masks, sector_colors, sector_names)):
        # Extend each segment by one point at the far boundary to close gaps between sectors
        indices = pos.index[mask].tolist()
        if i < 2:
            # append first point of next sector
            next_indices = pos.index[masks[i + 1]].tolist()
            if next_indices:
                indices = indices + [next_indices[0]]
        seg = pos.loc[indices, ('X', 'Y')].to_numpy()
        rotated_seg = rotate(seg, angle=track_angle)
        fig.add_trace(go.Scatter(
            x=rotated_seg[:, 0],
            y=rotated_seg[:, 1],
            mode='lines',
            line=dict(color=color, width=3),
            name=name,
            hoverinfo='skip',
        ))

    # Draw corners and labels.
    offset_vector = np.array([500.0, 0.0])
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner['Angle'] / 180 * np.pi
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        fig.add_trace(go.Scatter(
            x=[track_x, text_x],
            y=[track_y, text_y],
            mode='lines',
            line=dict(color='grey', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[text_x],
            y=[text_y],
            mode='markers+text',
            marker=dict(color='grey', size=18),
            text=[txt],
            textposition='middle center',
            textfont=dict(color='black', size=10),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text=session.event['Location'], x=0.5, font=dict(color='white', size=16)),
        legend=dict(font=dict(color='white', size=13)),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='black',
        paper_bgcolor='black',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )

    return fig

def plotdriverslaptimes(session, driver):
    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fpl.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    fig, ax = plt.subplots(figsize=(8, 8))

    for drv in driver:
        color = fpl.get_driver_color(drv, session=session)
        print(drv, color)
        driver_laps = session.laps.pick_drivers(drv).pick_quicklaps().reset_index()
        sns.scatterplot(data=driver_laps,
                        x="LapNumber",
                        y="LapTime",
                        ax=ax,
                        color=color,
                        s=80,
                        linewidth=0,
                        legend='auto')

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")

    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    ax.invert_yaxis()

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()

def plotsingledriverlaptimes(session, driver):
    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    f1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    fig, ax = plt.subplots(figsize=(8, 8))

    driver_laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index()
    sns.scatterplot(data=driver_laps,
                    x="LapNumber",
                    y="LapTime",
                    ax=ax,
                    hue="Compound", 
                    palette=f1.plotting.get_compound_mapping(session),
                    s=80,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")

    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    ax.invert_yaxis()

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()

def showraceresults(session):
    f1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers(drv)
        if len(drv_laps) == 0:
            continue
        abb = drv_laps['Driver'].iloc[0]
        style = f1.plotting.get_driver_style(identifier=abb,
                                             style=['color', 'linestyle'],
                                                session=session)
        ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
                label=abb, **style)
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')   
    ax.legend(bbox_to_anchor=(1.0, 1.02))
    plt.tight_layout()
    return fig

def tyreStrategies(session):
    laps = session.laps
    drivers = session.drivers
    drivers = [session.get_driver(driver)['Abbreviation'] for driver in drivers]
    stints = laps[['Driver', 'Stint', 'Compound', 'LapNumber']]
    stints = stints.groupby(['Driver', 'Stint', 'Compound'])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={'LapNumber': 'StintLength'})
    fig, ax = plt.subplots(figsize=(5, 10))

    for driver in drivers:
        driver_stints = stints[stints['Driver'] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            compound_color = f1.plotting.get_compound_color(row['Compound'], session=session)
            plt.barh(
                y=driver,
                width=row['StintLength'],
                left=previous_stint_end,
                color=compound_color,
                edgecolor='black',
                fill=True
            )
            previous_stint_end += row['StintLength']
    
    plt.xlabel('Lap Number')
    plt.grid(False)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig
            

def showqualifyingdeltas(session, drv_list=None):
    if drv_list is None:
        drv_list = []
    drivers = pd.unique(session.laps['Driver'])
    if drv_list:
        drivers = [d for d in drivers if d in drv_list]
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps) \
        .sort_values(by='LapTime') \
        .reset_index(drop=True)
    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        color = fpl.get_team_color(lap['Team'], session=session)
        team_colors.append(color)
    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')
    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} {session.name}\n"
                f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    return fig

@st.cache_data
def get_event_driver_abbreviations(year, event, session_name='Race'):
    """Return driver abbreviations using only the session results (no telemetry/laps loaded)."""
    try:
        sess = f1.get_session(year, event, session_name)
        sess.load(laps=False, telemetry=False, weather=False, messages=False)
        if hasattr(sess, 'results') and 'Abbreviation' in sess.results.columns:
            abbs = sess.results['Abbreviation'].dropna().unique().tolist()
            if abbs:
                return abbs
    except Exception as e:
        print(f"Could not get driver list for {event} {year} {session_name}: {e}")
    return []

def driverlaptimes(session):
    fpl.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    point_finishers = session.drivers[:10]
    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()

    finishing_order = [session.get_driver(i)['Abbreviation'] for i in point_finishers]
    fig, ax = plt.subplots(figsize=(10, 5))
    driver_laps['LapTime(s)'] = driver_laps['LapTime'].dt.total_seconds()
    sns.violinplot(data=driver_laps,
                   x='Driver',
                   y='LapTime(s)',
                   hue='Driver',
                   inner=None,
                   density_norm='area',
                   order=finishing_order,
                   palette=fpl.get_driver_color_mapping(session=session)
    )
    
    compound_mapping = fpl.get_compound_mapping(session=session)
    driver_laps = driver_laps[driver_laps['Compound'].isin(compound_mapping.keys())]
    hue_order = [c for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET',
                              'SUPERSOFT', 'ULTRASOFT', 'HYPERSOFT']
                 if c in driver_laps['Compound'].values]
    sns.swarmplot(data=driver_laps,
                  x='Driver',
                  y='LapTime(s)',
                  hue='Compound',
                  order=finishing_order,
                  palette=compound_mapping,
                  hue_order=hue_order,
                  linewidth=0,
                  size=4
    )
    ax.set_ylabel('Lap Time (s)')
    ax.set_xlabel('Driver')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return fig

# --- Data collection (vectorised and compact) -------------------------------
# drivers and raw dataframes
def showSectorTimesComparison(session, selected_driver1, selected_driver2):
    laps = session.laps
    drv1 = laps[(laps['Driver'] == selected_driver1) & (~laps.get('Deleted', False))]
    drv2 = laps[(laps['Driver'] == selected_driver2) & (~laps.get('Deleted', False))]

    drv1_row, drv1_mins = fastest_and_mins(drv1)
    drv2_row, drv2_mins = fastest_and_mins(drv2)

    drv1_team = drv1_row.get('Team', '') if drv1_row is not None else ''
    drv2_team = drv2_row.get('Team', '') if drv2_row is not None else ''

    # build ordered LapPart list and values (programmatic)
    lap_parts = [
        'Team',
        'FL Sector 1', 'FL Sector 2', 'FL Sector 3', 'FL Lap',
        'TL Sector 1', 'TL Sector 2', 'TL Sector 3',
        'Theoretical Lap'
    ]

    # helper to extract value (fastest lap sectors vs mins)
    def get_values_for_driver(driver_row, mins_dict, team):
        vals = [
            team,
            driver_row.get('Sector1Time', pd.NaT) if driver_row is not None else pd.NaT,
            driver_row.get('Sector2Time', pd.NaT) if driver_row is not None else pd.NaT,
            driver_row.get('Sector3Time', pd.NaT) if driver_row is not None else pd.NaT,
            driver_row.get('LapTime', pd.NaT) if driver_row is not None else pd.NaT,
            mins_dict.get('Sector1Time', pd.NaT),
            mins_dict.get('Sector2Time', pd.NaT),
            mins_dict.get('Sector3Time', pd.NaT),
            mins_dict.get('Sector1Time', pd.NaT) + mins_dict.get('Sector2Time', pd.NaT) + mins_dict.get('Sector3Time', pd.NaT)
        ]
        return vals

    vals1 = get_values_for_driver(drv1_row, drv1_mins, drv1_team)
    vals2 = get_values_for_driver(drv2_row, drv2_mins, drv2_team)

    data = {'LapPart': lap_parts, selected_driver1: vals1, selected_driver2: vals2}
    df = pd.DataFrame(data).set_index('LapPart')

    # --- Compute diff only for time rows ----------------------------------------
    time_rows = ['FL Sector 1','FL Sector 2','FL Sector 3','FL Lap', 'TL Sector 1','TL Sector 2','TL Sector 3', 'Theoretical Lap']

    # Vectorised safe conversion to timedelta, errors -> NaT
    a = pd.to_timedelta(df.loc[time_rows, selected_driver1], errors='coerce')
    b = pd.to_timedelta(df.loc[time_rows, selected_driver2], errors='coerce')

    # diff in seconds (or *1000 for ms)
    df.loc[time_rows, f'{selected_driver1}/{selected_driver2}'] = (a - b).dt.total_seconds()
    # keep non-time rows untouched (Team row remains string)
    # optional: fill NaN with '' for display
    df_display = df.copy().fillna('')
    df_display.loc[time_rows, :] = df_display.loc[time_rows, :].where(df_display.loc[time_rows, :].notna(), pd.NA)

    # --- Transpose and format display values -----------------------------------
    df2 = df_display.transpose()  # drivers become rows
    # format time rows to strings (safe)
    format_map = {}
    for col in ['FL Sector 1','FL Sector 2','FL Sector 3','TL Sector 1','TL Sector 2','TL Sector 3']:
        # use sector format
        df2[col] = df2[col].apply(lambda x: strftimedelta(x, '%s.%ms') if pd.notna(x) and not isinstance(x, (int,float)) else (f'{x:.3f}' if isinstance(x, (int,float)) else ''))

    # Format FL Lap / Theoretical Lap to mm:ss.ms if timedelta
    for col in ['FL Lap', 'Theoretical Lap']:
        if col in df2.columns:
            df2[col] = df2[col].apply(lambda x: strftimedelta(x, '%m:%s.%ms') if pd.notna(x) and not isinstance(x, (int,float)) else (f'{x:.3f}' if isinstance(x,(int,float)) else ''))

    # --- Build driver colour map (use Team cell from df2) -----------------------
    driver_colors = {}
    for drv in df2.index:
        team = df2.loc[drv, 'Team'] if 'Team' in df2.columns else None
        if team:
            try:
                driver_colors[drv] = fpl.get_team_color(team, session=session)
            except Exception:
                driver_colors[drv] = '#CCCCCC'
        else:
            driver_colors[drv] = '#CCCCCC'

    # --- (Optional) prepare styled Pandas Styler for notebook or HTML export ----
    def style_row(row):
        c = driver_colors.get(row.name, '#ffffff')
        txt = '#000' if sum(map(lambda v: int(v*255), to_rgb(c))) / 3 > 128 else '#fff'  # readable text
        return [f'background-color: {c}; color: {txt};' for _ in row]

    sector_header_styles = [
        {'selector': 'th', 'props': [('background-color', '#222'), ('color', '#fff')]},
        # S1 columns (blue)
        {'selector': 'th.col_heading.col1', 'props': [('background-color', '#4FC3F7'), ('color', '#000')]},
        {'selector': 'th.col_heading.col5', 'props': [('background-color', '#4FC3F7'), ('color', '#000')]},
        # S2 columns (yellow)
        {'selector': 'th.col_heading.col2', 'props': [('background-color', '#FFF176'), ('color', '#000')]},
        {'selector': 'th.col_heading.col6', 'props': [('background-color', '#FFF176'), ('color', '#000')]},
        # S3 columns (purple)
        {'selector': 'th.col_heading.col3', 'props': [('background-color', '#CE93D8'), ('color', '#000')]},
        {'selector': 'th.col_heading.col7', 'props': [('background-color', '#CE93D8'), ('color', '#000')]},
    ]
    styled = df2.style.apply(style_row, axis=1).set_table_styles(
        sector_header_styles
    ).set_properties(**{'border': '1px solid #999', 'padding': '6px'})
    return styled
    

def fastestlapstable(session):
    drivers = pd.unique(session.laps['Driver'])
    rows = []
    for drv in drivers:
        fastest = session.laps.pick_drivers(drv).pick_fastest()
        if fastest is None or pd.isna(fastest['LapTime']):
            continue
        rows.append({
            'Driver': drv,
            'Team': fastest['Team'],
            'Lap': int(fastest['LapNumber']),
            'Lap Time': strftimedelta(fastest['LapTime'], '%m:%s.%ms'),
            'Compound': fastest['Compound'],
        })
    df = pd.DataFrame(rows).sort_values('Lap Time').reset_index(drop=True)
    df.index += 1
    return df


def plot_driver_race_laps(session, driver):
    """Plotly scatter of all quick lap times for a driver, coloured by compound."""
    laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index(drop=True)
    if laps.empty:
        return None
    compound_colors = fpl.get_compound_mapping(session=session)
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()

    # Build y-axis ticks formatted as m:ss.ms
    y_min = laps['LapTime_s'].min()
    y_max = laps['LapTime_s'].max()
    tick_step = 0.5
    tickvals = [v / 2 for v in range(int(y_min * 2) - 1, int(y_max * 2) + 2)]
    ticktext = [f"{int(v // 60)}:{v % 60:05.2f}" for v in tickvals]

    fig = go.Figure()
    for compound, group in laps.groupby('Compound'):
        color = compound_colors.get(compound, '#AAAAAA')
        fig.add_trace(go.Scatter(
            x=group['LapNumber'],
            y=group['LapTime_s'],
            mode='markers',
            name=compound,
            marker=dict(color=color, size=8),
            customdata=group['LapTime'].apply(
                lambda x: strftimedelta(x, '%m:%s.%ms') if pd.notna(x) else ''
            ),
            hovertemplate='Lap %{x}<br>Time: %{customdata}<extra>' + compound + '</extra>',
        ))

    fig.update_layout(
        title=dict(text=f"{driver} — Lap Times", x=0.5, font=dict(color='white', size=14)),
        xaxis=dict(title='Lap Number', color='white', showgrid=False, dtick=5),
        yaxis=dict(
            title='Lap Time',
            color='white',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        legend=dict(font=dict(color='white')),
        plot_bgcolor='black',
        paper_bgcolor='black',
        margin=dict(l=0, r=0, t=50, b=0),
        height=350,
    )
    return fig


def marshal_sector_location(sector_num, circuit_info):
    """Return a human-readable location for a marshal sector, e.g. 'Between T3 & T4'."""
    try:
        ms = circuit_info.marshal_sectors
        corners = circuit_info.corners
        row = ms[ms['Number'] == sector_num]
        if row.empty:
            return ''
        dist = float(row.iloc[0]['Distance'])
        before = corners[corners['Distance'] <= dist]
        after = corners[corners['Distance'] > dist]

        def label(r):
            return f"T{int(r['Number'])}{r['Letter']}".rstrip()

        if not before.empty and not after.empty:
            return f"Between {label(before.iloc[-1])} & {label(after.iloc[0])}"
        elif not after.empty:
            return f"Before {label(after.iloc[0])}"
        elif not before.empty:
            return f"After {label(before.iloc[-1])}"
    except Exception:
        pass
    return ''


def showracedetails(year, race_name, session_name):
    session = load_session_cached(year, race_name, session_name)

    fig1 = showraceresults(session)
    st.pyplot(fig1); plt.close(fig1)

    st.subheader("Notable Events")
    msgs = session.race_control_messages
    if msgs is not None and not msgs.empty:
        notable = msgs[
            (msgs['Category'] == 'SafetyCar') |
            ((msgs['Category'] == 'Flag') & (msgs['Flag'].isin(['RED', 'CHEQUERED', 'DOUBLE YELLOW'])))
        ][['Lap', 'Category', 'Message']].reset_index(drop=True)
        if not notable.empty:
            try:
                circuit_info = session.get_circuit_info()
                def add_location(msg):
                    m = re.search(r'TRACK SECTOR\s+(\d+)', msg, re.IGNORECASE)
                    if m:
                        return marshal_sector_location(int(m.group(1)), circuit_info)
                    return ''
                notable['Location'] = notable['Message'].apply(add_location)
            except Exception:
                pass
            if len(notable) < 5:
                st.dataframe(notable, use_container_width=True, hide_index=True, height='content')
            else:
                st.dataframe(notable, use_container_width=True, hide_index=True, height=213)
        else:
            st.caption("No notable events recorded for this session.")
    else:
        st.caption("Race control messages not available for this session.")

    st.subheader("Fastest Laps")
    ft = fastestlapstable(session)
    event = st.dataframe(ft, use_container_width=True, on_select="rerun", selection_mode="single-row")
    selected_rows = event.selection.rows
    if selected_rows:
        driver = ft.iloc[selected_rows[0]]['Driver']
        fig = plot_driver_race_laps(session, driver)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    fig3 = driverlaptimes(session)
    st.pyplot(fig3); plt.close(fig3)

    fig2 = tyreStrategies(session)
    st.pyplot(fig2); plt.close(fig2)

def getSpeedTraceFor(session, driver1, driver2):
    driver1_lap = session.laps.pick_drivers(driver1).pick_fastest()
    driver2_lap = session.laps.pick_drivers(driver2).pick_fastest()
    try:
        driver1_tel = driver1_lap.get_car_data().add_distance()
        driver2_tel = driver2_lap.get_car_data().add_distance()
        circuit_info = session.get_circuit_info()
    except KeyError:
        return None

    # Calculate sector boundary distances from the overall fastest lap telemetry
    ref_lap = session.laps.pick_fastest()
    ref_tel = ref_lap.get_car_data().add_distance()
    s1_end_dist = ref_tel.loc[ref_tel['Time'] <= ref_lap['Sector1Time'], 'Distance'].max()
    s2_end_dist = ref_tel.loc[ref_tel['Time'] <= ref_lap['Sector1Time'] + ref_lap['Sector2Time'], 'Distance'].max()

    sector_colors = {1: '#4FC3F7', 2: '#FFF176', 3: '#CE93D8'}  # blue, yellow, purple

    def corner_sector(dist):
        if dist <= s1_end_dist:
            return 1
        elif dist <= s2_end_dist:
            return 2
        return 3

    d1_color = fpl.get_team_color(driver1_lap['Team'], session=session)
    d2_color = fpl.get_team_color(driver2_lap['Team'], session=session)
    d2_linestyle = '--' if d1_color == d2_color else '-'

    fig, ax = plt.subplots()
    ax.plot(driver1_tel['Distance'], driver1_tel['Speed'], color=d1_color, label=driver1)
    ax.plot(driver2_tel['Distance'], driver2_tel['Speed'], color=d2_color, label=driver2, linestyle=d2_linestyle)

    # Draw vertical dotted lines at each corner
    v_min = driver1_tel['Speed'].min()
    v_max = driver1_tel['Speed'].max()
    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20,
            linestyles='dotted', colors='grey')

    # Plot corner numbers coloured by sector (S1=blue, S2=yellow, S3=purple)
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        color = sector_colors[corner_sector(corner['Distance'])]
        ax.text(corner['Distance'], v_min-30, txt,
                va='center_baseline', ha='center', size='small', color=color)

    ax.set_xlabel('Distance in m')
    ax.set_ylabel('Speed in km/h')

    ax.legend()
    plt.suptitle(f"Fastest Lap Comparison \n "
                f"{session.event['EventName']} {session.event.year} {session.name}")
    return fig

@st.cache_data(ttl=3600)
def getdriverstandings(year, round):
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=year, round=round)
    return standings.content[0]

@st.cache_data(ttl=3600)
def calculatemaxpointsforremainingseason(year, round):
    POINTS_FOR_SPRINT = 8 + 25 # Winning the sprint and race
    POINTS_FOR_CONVENTIONAL = 25 # Winning the race

    events = f1.events.get_event_schedule(year, backend='ergast')
    events = events[events['RoundNumber'] > round]
    print(events[["RoundNumber","EventName","EventFormat"]])
    # Count how many sprints and conventional races are left
    sprint_events = len(events.loc[events["EventFormat"] == "sprint_qualifying"])
    conventional_events = len(events.loc[events["EventFormat"] == "conventional"])
    print(sprint_events, conventional_events)
    # Calculate points for each
    sprint_points = sprint_events * POINTS_FOR_SPRINT
    conventional_points = (sprint_events + conventional_events) * POINTS_FOR_CONVENTIONAL
    print(sprint_points, conventional_points)
    return sprint_points + conventional_points

def getSpeedDifferenceChart(session, driver1, driver2):
    """Plotly chart of (driver1 speed - driver2 speed) vs distance, coloured by who is faster."""
    driver1_lap = session.laps.pick_drivers(driver1).pick_fastest()
    driver2_lap = session.laps.pick_drivers(driver2).pick_fastest()
    # Sector boundary distances from overall fastest lap
    ref_lap = session.laps.pick_fastest()
    try:
        d1_tel = driver1_lap.get_car_data().add_distance()
        d2_tel = driver2_lap.get_car_data().add_distance()
        ref_tel = ref_lap.get_car_data().add_distance()
        circuit_info = session.get_circuit_info()
    except KeyError:
        return None
    s1_end_dist = ref_tel.loc[ref_tel['Time'] <= ref_lap['Sector1Time'], 'Distance'].max()
    s2_end_dist = ref_tel.loc[ref_tel['Time'] <= ref_lap['Sector1Time'] + ref_lap['Sector2Time'], 'Distance'].max()

    # Interpolate both onto a shared distance grid (driver1's grid as reference)
    dist = d1_tel['Distance'].to_numpy()
    d1_speed = d1_tel['Speed'].to_numpy()
    d2_speed_interp = np.interp(dist, d2_tel['Distance'].to_numpy(), d2_tel['Speed'].to_numpy())
    diff = d1_speed - d2_speed_interp

    d1_color = fpl.get_team_color(driver1_lap['Team'], session=session)
    d2_color = fpl.get_team_color(driver2_lap['Team'], session=session)

    def hex_to_rgba(hex_color, alpha=0.33):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    # Split diff into two series: where d1 is faster (positive) and d2 is faster (negative)
    diff_pos = np.where(diff >= 0, diff, 0)
    diff_neg = np.where(diff < 0, diff, 0)

    fig = go.Figure()

    # Shaded fill: driver1 faster
    fig.add_trace(go.Scatter(
        x=np.concatenate([dist, dist[::-1]]),
        y=np.concatenate([diff_pos, np.zeros(len(dist))]),
        fill='toself',
        fillcolor=hex_to_rgba(d1_color),
        line=dict(width=0),
        name=f'{driver1} faster',
        hoverinfo='skip',
    ))

    # Shaded fill: driver2 faster
    fig.add_trace(go.Scatter(
        x=np.concatenate([dist, dist[::-1]]),
        y=np.concatenate([diff_neg, np.zeros(len(dist))]),
        fill='toself',
        fillcolor=hex_to_rgba(d2_color),
        line=dict(width=0),
        name=f'{driver2} faster',
        hoverinfo='skip',
    ))

    # Main diff line
    fig.add_trace(go.Scatter(
        x=dist,
        y=diff,
        mode='lines',
        line=dict(color='white', width=1.5),
        name=f'{driver1} − {driver2}',
        hovertemplate='Distance: %{x:.0f} m<br>Δ Speed: %{y:.1f} km/h<extra></extra>',
    ))

    # Zero baseline
    fig.add_hline(y=0, line=dict(color='grey', width=1, dash='dot'))

    # Sector background bands
    track_max = float(dist.max())
    sector_bands = [
        (0, s1_end_dist, 'rgba(79,195,247,0.06)'),
        (s1_end_dist, s2_end_dist, 'rgba(255,241,118,0.06)'),
        (s2_end_dist, track_max, 'rgba(206,147,216,0.06)'),
    ]
    for x0, x1, color in sector_bands:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer='below')

    # Corner markers
    sector_colors = {1: '#4FC3F7', 2: '#FFF176', 3: '#CE93D8'}
    y_annot = float(np.min(diff)) - 5
    for _, corner in circuit_info.corners.iterrows():
        d = corner['Distance']
        sector = 1 if d <= s1_end_dist else (2 if d <= s2_end_dist else 3)
        fig.add_vline(x=d, line=dict(color='grey', width=1, dash='dot'))
        fig.add_annotation(
            x=d, y=y_annot,
            text=f"{corner['Number']}{corner['Letter']}",
            showarrow=False,
            font=dict(color=sector_colors[sector], size=9),
            yanchor='top',
        )

    fig.update_layout(
        title=dict(
            text=f"Speed Difference: {driver1} − {driver2}",
            x=0.5,
            font=dict(color='white', size=14),
        ),
        xaxis=dict(title='Distance (m)', color='white', showgrid=False),
        yaxis=dict(title='Speed Difference (km/h)', color='white', showgrid=True,
                   gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(font=dict(color='white')),
        plot_bgcolor='black',
        paper_bgcolor='black',
        margin=dict(l=0, r=0, t=50, b=0),
        height=350,
    )
    return fig


def driverComparison(year, selected_race, selected_session, selected_driver1, selected_driver2):
    fpl.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    session = load_session_cached(year, selected_race, selected_session)
    fig1 = getSpeedTraceFor(session, selected_driver1, selected_driver2)
    fig2 = showqualifyingdeltas(session, drv_list=[selected_driver1,selected_driver2])
    fig3 = getSpeedDifferenceChart(session, selected_driver1, selected_driver2)
    styled = showSectorTimesComparison(session, selected_driver1, selected_driver2)
    st.pyplot(fig2); plt.close(fig2)
    if fig1 is not None:
        st.pyplot(fig1); plt.close(fig1)
    else:
        st.warning("Speed trace data not available for this session.")
    if fig3 is not None:
        st.plotly_chart(fig3, width='stretch')
    else:
        st.warning("Speed difference chart data not available for this session.")
    html = styled.to_html()
    components.html(html, height=300, scrolling=True)

def get_driver_lap_times_df(session, driver):
    """Return a DataFrame of valid lap times for a driver with formatted times."""
    laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index(drop=True)
    df = laps[['LapNumber', 'LapTime', 'Compound', 'Stint']].copy()
    df['LapNumber'] = df['LapNumber'].astype(int)
    df['LapTime'] = df['LapTime'].apply(
        lambda x: strftimedelta(x, '%m:%s.%ms') if pd.notna(x) else ''
    )
    return df


def plot_lap_telemetry(session, driver, lap_numbers):
    """Plot speed traces for 1 or 2 specific laps of a driver."""
    fpl.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')
    circuit_info = session.get_circuit_info()
    driver_laps = session.laps.pick_drivers(driver)

    colors = ['#e8002d', '#00d2be']
    tels = []
    all_speeds = []
    for lap_num in lap_numbers:
        lap = driver_laps[driver_laps['LapNumber'] == lap_num].iloc[0]
        tel = lap.get_car_data().add_distance()
        tels.append(tel)
        all_speeds.extend(tel['Speed'].dropna().tolist())

    v_min = min(all_speeds)
    v_max = max(all_speeds)

    fig, ax = plt.subplots()
    for i, (lap_num, tel) in enumerate(zip(lap_numbers, tels)):
        ax.plot(tel['Distance'], tel['Speed'], color=colors[i], label=f'Lap {lap_num}')

    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min - 20, ymax=v_max + 20,
              linestyles='dotted', colors='grey')
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(corner['Distance'], v_min - 30, txt,
                va='center_baseline', ha='center', size='small')

    ax.set_xlabel('Distance in m')
    ax.set_ylabel('Speed in km/h')
    ax.legend()
    ax.set_title(f"{driver} — {session.event['EventName']} {session.event.year} {session.name}")
    plt.tight_layout()
    return fig


def calculatewhocanwin(driver_standings, max_points):
    LEADER_POINTS = int(driver_standings.loc[0]['points'])

    for i, _ in enumerate(driver_standings.iterrows()):
        driver = driver_standings.loc[i]
        driver_max_points = int(driver["points"]) + max_points
        can_win = 'No' if driver_max_points < LEADER_POINTS else 'Yes'

        st.write(f"{driver['position']}: {driver['givenName'] + ' ' + driver['familyName']}, "
              f"Current points: {driver['points']}, "
              f"Theoretical max points: {driver_max_points}, "
              f"Can win: {can_win}")
        
# Maps Ergast constructorId -> (hex colour, display name)
_CONSTRUCTOR_COLORS = {
    'red_bull':     ('#3671C6', 'Red Bull'),
    'mercedes':     ('#27F4D2', 'Mercedes'),
    'ferrari':      ('#E8002D', 'Ferrari'),
    'mclaren':      ('#FF8000', 'McLaren'),
    'aston_martin': ('#358C75', 'Aston Martin'),
    'alpine':       ('#FF87BC', 'Alpine'),
    'williams':     ('#64C4FF', 'Williams'),
    'rb':           ('#6692FF', 'RB'),
    'alphatauri':   ('#5E8FAA', 'AlphaTauri'),
    'haas':         ('#B6BABD', 'Haas'),
    'sauber':       ('#52E252', 'Kick Sauber'),
    'alfa':         ('#C92D4B', 'Alfa Romeo'),
    'racing_point': ('#F596C8', 'Racing Point'),
    'renault':      ('#FFF500', 'Renault'),
    'toro_rosso':   ('#469BFF', 'Toro Rosso'),
    'force_india':  ('#F596C8', 'Force India'),
    'lotus_f1':     ('#FFB800', 'Lotus'),
    'manor':        ('#C0392B', 'Manor'),
    'caterham':     ('#1E5B23', 'Caterham'),
    'marussia':     ('#E04B09', 'Marussia'),
}

@st.cache_data(ttl=3600)
def showteamstanding(year, round):
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    results = []

    for rnd, race in races['raceName'].items():
        temp = ergast.get_race_results(season=year, round=rnd + 1)
        try:
            race_df = temp.content[0][['constructorId', 'points']].copy()
        except:
            break

        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            sprint_df = sprint.content[0][['constructorId', 'points']].copy()
            race_df = pd.concat([race_df, sprint_df])

        team_pts = race_df.groupby('constructorId')['points'].sum().reset_index()
        team_pts['round'] = rnd + 1
        team_pts['race'] = race.removesuffix(' Grand Prix')
        results.append(team_pts)

    results = pd.concat(results)
    race_name_map = results[['round', 'race']].drop_duplicates().set_index('round')['race']

    team_results = results.pivot_table(index='constructorId', columns='round', values='points', aggfunc='sum').fillna(0)
    team_results['total_points'] = team_results.sum(axis=1)
    team_results = team_results.sort_values('total_points', ascending=False)

    round_cols = [c for c in team_results.columns if c != 'total_points']
    team_results = team_results.rename(columns={r: race_name_map[r] for r in round_cols})
    team_results = team_results.rename(columns={'total_points': 'Total'})

    constructors = team_results.index.tolist()
    display_names = [_CONSTRUCTOR_COLORS.get(c, ('#AAAAAA', c.replace('_', ' ').title()))[1] for c in constructors]
    team_colors_list = [_CONSTRUCTOR_COLORS.get(c, ('#AAAAAA', ''))[0] for c in constructors]

    race_cols = [c for c in team_results.columns if c != 'Total']
    race_z = team_results[race_cols].values
    total_z = team_results[['Total']].values

    blue_scale = [
        [0,    'rgb(198, 219, 239)'],
        [0.25, 'rgb(107, 174, 214)'],
        [0.5,  'rgb(33,  113, 181)'],
        [0.75, 'rgb(8,   81,  156)'],
        [1,    'rgb(8,   48,  107)'],
    ]

    num_races = len(race_cols)
    total_width = 1 / (num_races + 1)
    race_width = 1 - total_width - 0.01

    fig = make_subplots(rows=1, cols=2, column_widths=[race_width, total_width], horizontal_spacing=0.01)

    fig.add_trace(go.Heatmap(
        z=race_z,
        x=race_cols,
        y=display_names,
        text=race_z,
        texttemplate='%{text}',
        colorscale=blue_scale,
        showscale=False,
        hovertemplate='Team: %{y}<br>Race: %{x}<br>Points: %{z}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=total_z,
        x=['Total'],
        y=display_names,
        text=total_z,
        texttemplate='%{text}',
        colorscale=blue_scale,
        showscale=False,
        hovertemplate='Team: %{y}<br>Total: %{z}<extra></extra>',
    ), row=1, col=2)

    fig.update_xaxes(side='top', showgrid=False, showline=False, title_text='')
    fig.update_yaxes(tickmode='linear', showgrid=True, gridwidth=1,
                     gridcolor='LightGrey', showline=False, tickson='boundaries',
                     title_text='', autorange='reversed', showticklabels=False)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Team name labels coloured by team — Plotly doesn't support per-tick colours
    # so we use annotations positioned just left of the plot area
    for name, color in zip(display_names, team_colors_list):
        fig.add_annotation(
            x=0, y=name,
            xref='paper', yref='y',
            text=f'<b>{name}</b>',
            showarrow=False,
            xanchor='right',
            font=dict(color=color, size=11),
        )

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=110, r=0, b=0, t=0))
    return fig


@st.cache_data(ttl=3600)
def showdriverstanding(year, round):
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        # Get results. Note that we use the round no. + 1, because the round no.
        # starts from one (1) instead of zero (0)
        temp = ergast.get_race_results(season=year, round=rnd + 1)
        try:
            temp = temp.content[0]
        except:
            break

        # If there is a sprint, get the results as well
        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            temp = pd.merge(temp, sprint.content[0], on='driverCode', how='left')
            # Add sprint points and race points to get the total
            temp['points'] = temp['points_x'] + temp['points_y']
            temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp = temp[['round', 'race', 'driverCode', 'points']]  # Keep useful cols.
        results.append(temp)

    # Append all races into a single dataframe
    results = pd.concat(results)
    races = results['race'].drop_duplicates()

    results = results.pivot(index='driverCode', columns='round', values='points')
    # Here we have a 22-by-22 matrix (22 races and 22 drivers, incl. DEV and HUL)

    # Rank the drivers by their total points
    results['total_points'] = results.sum(axis=1)
    results = results.sort_values(by='total_points', ascending=False)

    # Use race name, instead of round no., as column names; keep Total on the right
    round_cols = [c for c in results.columns if c != 'total_points']
    results = results.rename(columns=dict(zip(round_cols, races)))
    results = results.rename(columns={'total_points': 'Total'})

    blue_scale = [
        [0,    'rgb(198, 219, 239)'],
        [0.25, 'rgb(107, 174, 214)'],
        [0.5,  'rgb(33,  113, 181)'],
        [0.75, 'rgb(8,   81,  156)'],
        [1,    'rgb(8,   48,  107)'],
    ]

    race_cols = [c for c in results.columns if c != 'Total']
    drivers = results.index.tolist()
    race_z = results[race_cols].values
    total_z = results[['Total']].values

    num_races = len(race_cols)
    # Give Total column roughly the same width as one race column
    total_width = 1 / (num_races + 1)
    race_width = 1 - total_width - 0.01  # small gap between panels

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[race_width, total_width],
        horizontal_spacing=0.01,
    )

    fig.add_trace(go.Heatmap(
        z=race_z,
        x=race_cols,
        y=drivers,
        text=race_z,
        texttemplate='%{text}',
        colorscale=blue_scale,
        showscale=False,
        hovertemplate='Driver: %{y}<br>Race: %{x}<br>Points: %{z}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=total_z,
        x=['Total'],
        y=drivers,
        text=total_z,
        texttemplate='%{text}',
        colorscale=blue_scale,
        showscale=False,
        hovertemplate='Driver: %{y}<br>Total: %{z}<extra></extra>',
    ), row=1, col=2)

    fig.update_xaxes(side='top', showgrid=False, showline=False, title_text='')
    fig.update_yaxes(tickmode='linear', showgrid=True, gridwidth=1,
                     gridcolor='LightGrey', showline=False, tickson='boundaries',
                     title_text='', autorange='reversed')
    fig.update_yaxes(showticklabels=False, row=1, col=2)  # no duplicate driver labels
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0))

    return fig


def _build_podium_figure(predictions: pd.DataFrame, driver_colors: dict) -> go.Figure:
    """
    Build a Plotly figure showing the predicted podium as classic stepped blocks.
    P2 left (medium height), P1 centre (tallest), P3 right (shortest).
    Driver names are rendered in their team colour.
    """
    order = [1, 0, 2]  # P2, P1, P3 column order for classic podium look
    step_heights = [0.70, 1.0, 0.50]
    x_positions = [0, 1, 2]
    labels = ['P2', 'P1', 'P3']

    fig = go.Figure()

    for col_idx, (rank_idx, height, x, label) in enumerate(
            zip(order, step_heights, x_positions, labels)):
        if rank_idx >= len(predictions):
            continue

        row = predictions.iloc[rank_idx]
        driver = row['driverCode']
        color = driver_colors.get(driver, '#AAAAAA')
        podium_prob = row['podium_prob']
        win_prob = row['win_prob']

        # Podium step block
        fig.add_shape(
            type='rect',
            x0=x - 0.38, x1=x + 0.38,
            y0=0, y1=height,
            fillcolor=color,
            opacity=0.85,
            line=dict(color='white', width=1),
        )

        # Position label (P1/P2/P3) inside block
        fig.add_annotation(
            x=x, y=height / 2,
            text=f'<b>{label}</b>',
            showarrow=False,
            font=dict(size=22, color='white'),
            xanchor='center', yanchor='middle',
        )

        # Driver name above block
        fig.add_annotation(
            x=x, y=height + 0.06,
            text=f'<b>{driver}</b>',
            showarrow=False,
            font=dict(size=15, color=color),
            xanchor='center', yanchor='bottom',
        )

        # Probability badges below block
        fig.add_annotation(
            x=x, y=-0.12,
            text=f'Win {win_prob*100:.0f}%  |  Podium {podium_prob*100:.0f}%',
            showarrow=False,
            font=dict(size=11, color='#CCCCCC'),
            xanchor='center', yanchor='top',
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.6, 2.6]),
        yaxis=dict(visible=False, range=[-0.25, 1.25]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=50),
        height=320,
        showlegend=False,
    )
    return fig


def _show_race_prediction():
    """Render the Race Prediction expander contents."""
    st.subheader("Next Race Podium Prediction")
    st.caption(
        "Trained on the last 3 seasons of race results. "
        "Current season data is heavily weighted to account for regulation changes.\n"
        "For fun only — not fantasy f1 advice!"
    )

    # ---- Sidebar controls for this section ----
    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        reg_boost = st.slider(
            "Current season weight boost (regulation change factor)",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            help="Higher = trust this season's results more over prior seasons. "
                 "Set higher if there was a big regulation change.",
        )
    with col_cfg2:
        num_seasons = st.selectbox("Training seasons", [1, 2, 3], index=2)

    # ---- Identify next race ----
    current_year = predictor.CURRENT_YEAR
    with st.spinner("Finding next race..."):
        next_race = predictor.get_next_race_info(current_year)

    if next_race is None:
        st.info("No upcoming races found for this season.")
        return

    st.markdown(
        f"**Next race:** {next_race['name']} — "
        f"{next_race['location']}, {next_race['country']} "
        f"({next_race['date']})"
    )

    # ---- Weather ----
    days_away = (next_race['date'] - __import__('datetime').date.today()).days
    weather = predictor.fetch_race_weather(
        next_race.get('lat'), next_race.get('lon'), next_race['date']
    )

    if weather is not None:
        wet_flag = weather['is_wet']
        rain_icon = "🌧" if wet_flag else "☀️"
        st.markdown(f"**Weather forecast:** {rain_icon} {weather['description']}")
        wet_override = st.toggle(
            "Override: treat as wet race",
            value=wet_flag,
            help="Toggle if you want to see how wet conditions change the prediction.",
        )
        wet_condition = wet_override
    else:
        if days_away > 16:
            st.caption(f"Race is {days_away} days away — beyond reliable forecast window.")
        wet_condition = st.toggle(
            "Wet race conditions?",
            value=False,
            help="Enable to factor in wet-weather conditions for the prediction.",
        )

    # ---- Grid position input ----
    st.markdown("**Expected grid positions** (edit if qualifying has happened):")

    # Get current drivers from the results this season
    with st.spinner("Training model..."):
        podium_model, win_model, results_df, sw = predictor.run_prediction_pipeline(
            current_year, num_seasons, reg_boost
        )

    if podium_model is None:
        st.error("Not enough historical data to train the model.")
        return

    current_drivers = (
        results_df[results_df['year'] == current_year]['driverCode']
        .unique().tolist()
    )
    # Default: sort by championship points (best first = grid P1 proxy)
    current_season = results_df[results_df['year'] == current_year]
    driver_pts_total = (current_season.groupby('driverCode')['points']
                                      .sum()
                                      .sort_values(ascending=False))
    ordered_drivers = driver_pts_total.index.tolist()
    # Any driver not yet in results gets appended
    for d in current_drivers:
        if d not in ordered_drivers:
            ordered_drivers.append(d)

    # Show a compact grid input table
    default_grid = {drv: i + 1 for i, drv in enumerate(ordered_drivers)}

    with st.expander("Edit expected grid positions", expanded=False):
        grid_cols = st.columns(4)
        user_grid = {}
        for i, drv in enumerate(ordered_drivers):
            with grid_cols[i % 4]:
                pos = st.number_input(
                    drv, min_value=1, max_value=len(ordered_drivers),
                    value=default_grid[drv], key=f"grid_{drv}",
                )
                user_grid[drv] = pos

    driver_grid_map = user_grid if user_grid else default_grid

    # ---- Run prediction ----
    next_race_features = predictor.build_next_race_features(
        results_df, next_race, current_year, driver_grid_map, wet_condition
    )

    predictions = predictor.predict_podium(podium_model, win_model, next_race_features)

    # ---- Get team colours ----
    # Build driver -> hex colour from FastF1 if possible
    driver_colors = {}
    try:
        last_round = int(results_df[results_df['year'] == current_year]['round'].max())
        sess = f1.get_session(current_year, last_round, 'R')
        sess.load(laps=False, telemetry=False, weather=False, messages=False)
        for drv in predictions['driverCode']:
            try:
                driver_colors[drv] = fpl.get_driver_color(drv, session=sess)
            except Exception:
                driver_colors[drv] = '#AAAAAA'
    except Exception:
        pass

    # ---- Podium visual ----
    st.markdown("### Predicted Podium")
    podium_fig = _build_podium_figure(predictions.head(3), driver_colors)
    st.plotly_chart(podium_fig, use_container_width=True)

    # ---- Full probability bar chart ----
    st.markdown("### Win Probability — All Drivers")
    bar_df = predictions.copy()
    bar_df['color'] = bar_df['driverCode'].map(
        lambda d: driver_colors.get(d, '#AAAAAA')
    )
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_df['driverCode'],
        y=(bar_df['win_prob'] * 100).round(1),
        marker_color=bar_df['color'].tolist(),
        text=(bar_df['win_prob'] * 100).round(1).astype(str) + '%',
        textposition='outside',
        hovertemplate='%{x}<br>Win: %{y:.1f}%<extra></extra>',
        name='Win probability',
    ))
    fig_bar.add_trace(go.Bar(
        x=bar_df['driverCode'],
        y=(bar_df['podium_prob'] * 100).round(1),
        marker_color=bar_df['color'].tolist(),
        opacity=0.35,
        hovertemplate='%{x}<br>Podium: %{y:.1f}%<extra></extra>',
        name='Podium probability',
        showlegend=True,
    ))
    fig_bar.update_layout(
        barmode='overlay',
        xaxis_title='Driver',
        yaxis_title='Probability (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Feature importances ----
    with st.expander("Model feature importances (win model)", expanded=False):
        imp_df = predictor.get_feature_importances(win_model)
        fig_imp = go.Figure(go.Bar(
            x=imp_df['importance'],
            y=imp_df['feature'],
            orientation='h',
            marker_color='#4FC3F7',
        ))
        fig_imp.update_layout(
            xaxis_title='Importance',
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("**Model**")
        st.markdown(
            "Gradient Boosting — an ensemble of 200 shallow decision trees built sequentially, "
            "each one correcting the errors of the previous. Two separate classifiers are trained: "
            "one for **win** (P1) and one for **podium** (top 3). Win probability is used to order "
            "the podium steps; podium probability is shown as the wider bar in the chart above."
        )

        st.markdown("**Setup**")
        st.code(
            "n_estimators=200  # trees in the ensemble\n"
            "learning_rate=0.05  # each tree contributes 5% of its correction\n"
            "max_depth=3  # shallow trees — at most 8 leaves\n"
            "subsample=0.8  # each tree sees 80% of rows (reduces overfitting)",
            language="python",
        )
        st.markdown(
            "Current season races are weighted up to **~15× more** than races from two seasons ago, "
            "combining the regulation-change boost with a within-season recency factor."
        )

        st.markdown("**Limitations**")
        st.markdown(
            "- **Small dataset** — ~1,400 rows across 3 seasons. The model is directional, not precise.\n"
            "- **Grid position** — before qualifying, grid is estimated from championship order, which blurs predictions significantly. After qualifying it becomes much more reliable.\n"
            "- **No intra-race strategy** — safety cars, pit windows, and tyre deg aren't modelled.\n"
            "- **Wet is binary** — a damp track and a monsoon are treated the same."
        )


st.set_page_config(layout="wide")
st.title("F1 Dashboard - FastF1")

_pic_cols = st.columns(4)
for _col, _img in zip(_pic_cols, sorted(os.listdir("pics"))):
    _col.image(f"pics/{_img}", use_container_width=True)

st.sidebar.header("F1 Controls")

if st.sidebar.button("Clear Cache", help="Free memory by clearing all cached session data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

year = st.sidebar.slider("Select Year", min_value=2015, max_value=2026, value=2026)
schedule = getschedule(year)

race_names = schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()
selected_race = st.sidebar.selectbox("Select Race", race_names)
race_info = schedule[schedule['EventName'] == selected_race].iloc[0]
round = int(race_info['RoundNumber'])

#if st.sidebar.button("Driver Standings"):
with st.expander("Driver Standings", expanded=False):
    # Get the current drivers standings
    fig = showdriverstanding(year, round)
    st.plotly_chart(fig, width='stretch')

with st.expander("Team Standings", expanded=False):
    fig = showteamstanding(year, round)
    st.plotly_chart(fig, width='stretch')

with st.expander("Championship", expanded=False):
    driver_standings = getdriverstandings(year, round)
    points = calculatemaxpointsforremainingseason(year, round)
    calculatewhocanwin(driver_standings, points)

with st.expander("Next Race Prediction", expanded=False):
    _show_race_prediction()

st.write(f"You selected: {year} - {selected_race}")
st.subheader("Selection Details")
st.write(f"Year: {year}")
st.write(f"Country: {race_info['Country']}")
st.write(f"Date: {str(race_info['EventDate'])}")
st.write(f"Round: {round}")
st.write(f"Race: {selected_race}")

if year >= 2018:
    try:
        weather_session = load_session_cached(year, selected_race, "R")
        wd = weather_session.weather_data
        if wd is not None and not wd.empty:
            avg_air = wd['AirTemp'].mean()
            avg_track = wd['TrackTemp'].mean()
            avg_humidity = wd['Humidity'].mean()
            avg_wind = wd['WindSpeed'].mean()
            rainfall = wd['Rainfall'].any()
            st.write(f"**Weather (Race):** Air {avg_air:.1f}°C | Track {avg_track:.1f}°C | "
                     f"Humidity {avg_humidity:.0f}% | Wind {avg_wind:.1f} km/h | "
                     f"{'🌧 Rain' if rainfall else 'Dry'}")
    except Exception:
        pass

if year < 2018:
    st.info("Detailed session data is not available for seasons prior to 2018. Only Driver and Team Standings are shown.")
else:
    with st.expander("Track Map"):
        session = load_session_cached(year, selected_race, "FP1")
        fig = drawtrackfor(session)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Track map data not available for this session.")

    sessions = []
    for col in schedule.columns:
        if col.startswith("Session") and not col.endswith("Date") and not col.endswith("DateUtc"):
            val = race_info[col]
            if pd.notna(val):
                sessions.append(val)
    selected_session = st.sidebar.selectbox("Select Session", sessions)

    driver_codes = get_event_driver_abbreviations(year, selected_race, selected_session)
    selected_driver1 = st.sidebar.selectbox("Select Driver", driver_codes)
    selected_driver2 = st.sidebar.selectbox("Select Comparison Driver", driver_codes)

    with st.expander("Session Overview"):
        showracedetails(year, selected_race, selected_session)

    with st.expander("Lap Comparison"):
        session = load_session_cached(year, selected_race, selected_session)
        lap_df = get_driver_lap_times_df(session, selected_driver1)
        st.subheader(f"{selected_driver1} — Lap Times")
        st.dataframe(lap_df, hide_index=True, width='stretch')

        lap_options = lap_df['LapNumber'].tolist()
        selected_laps = st.multiselect(
            "Select 1 or 2 laps to compare",
            options=lap_options,
            max_selections=2,
            format_func=lambda n: f"Lap {n}",
            key='compare_laps_selection',
        )

        if selected_laps and st.button("Show Lap Telemetry"):
            st.session_state.compare_laps_tel_laps = selected_laps

        if st.session_state.get('compare_laps_tel_laps'):
            fig = plot_lap_telemetry(session, selected_driver1,
                                     st.session_state.compare_laps_tel_laps)
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Driver Comparison"):
        driverComparison(year, selected_race, selected_session, selected_driver1, selected_driver2)

st.info("F1 Dashboard\nBuilt by Steve Johnstone 2026\n")
st.info("F1 is a registered trademark of Formula One Licensing B.V. All data sourced from Ergast API and FastF1.")
st.info("This dashboard is for educational purposes only and does not represent the views of Formula One or any affiliated parties.")
