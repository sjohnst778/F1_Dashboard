import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
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

fastf1.Cache.enable_cache('f1cache')

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
        return strftimedelta(x, fmt_seconds if 'Sector' in cur_col else fmt_lap)
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


@st.cache_resource
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
    fpl.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    circuit_info = session.get_circuit_info()
    fig, ax = plt.subplots()
    # Get an array of shape [n, 2] where n is the number of points and the second
    # axis is x and y.
    track = pos.loc[:, ('X', 'Y')].to_numpy()

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate and plot the track map.
    rotated_track = rotate(track, angle=track_angle)
    ax.plot(rotated_track[:, 0], rotated_track[:, 1],color='white', linewidth=1.2)

    offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y
        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)
        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Draw a circle next to the track.
        ax.scatter(text_x, text_y, color='grey', s=140)
        # Draw a line from the track to this circle.
        ax.plot([track_x, text_x], [track_y, text_y], color='grey')
        # Finally, print the corner number inside the circle.
        ax.text(text_x, text_y, txt,
                 va='center_baseline', ha='center', size='small', color='white', zorder=6)

    ax.set_title(session.event['Location'])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
#    plt.show()
    return fig

def plotdriverslaptimes(session, driver):
    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fpl.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
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
    f1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
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
    """Return a list of 3-letter driver abbreviations for the given event/session.
    - year: int
    - event: the event identifier from the schedule (try EventName or EventShortName)
    - session_name: typically 'Race' or 'Qualifying'

    This function prefers to read the abbreviations from session.laps['Driver'] (most reliable),
    then falls back to session.results['Abbreviation'], and finally to session.get_driver().
    """
    try:
        sess = f1.get_session(year, event, session_name)
    except Exception as e:
        print(f"Could not get session {session_name} for {event} {year}: {e}")
        return []

    # Load minimal data (avoid full telemetry). Some FastF1 versions accept telemetry/weather args.
    try:
        sess.load(telemetry=False, weather=False)
    except TypeError:
        sess.load()

    # 1) Prefer abbreviations from the laps table (this commonly contains 3-letter codes)
    try:
        if hasattr(sess, 'laps') and len(sess.laps) > 0:
            # session.laps['Driver'] is usually the 3-letter code
            abbs = pd.unique(sess.laps['Driver']).tolist()
            return [str(a) for a in abbs]
    except Exception:
        pass

    # 2) Fallback: check a results table if present
    try:
        if hasattr(sess, 'results') and 'Abbreviation' in sess.results.columns:
            return sess.results['Abbreviation'].dropna().unique().tolist()
    except Exception:
        pass

    # 3) Last resort: query session.drivers + session.get_driver and extract 'Abbreviation'
    abbs = []
    for drv in sess.drivers:
        try:
            info = sess.get_driver(drv)
        except Exception:
            info = None
        abb = None
        if isinstance(info, dict):
            abb = info.get('Abbreviation') or info.get('abbr') or info.get('Abbrev')
        elif hasattr(info, 'get'):
            try:
                abb = info.get('Abbreviation') or info.get('abbr')
            except Exception:
                abb = None
        # fallback to driver identifier
        if not abb:
            abb = str(drv)
        abbs.append(abb)
    return abbs

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
    
    sns.swarmplot(data=driver_laps,
                  x='Driver',
                  y='LapTime(s)',
                  hue='Compound',
                  order=finishing_order,
                  palette=fpl.get_compound_mapping(session=session),
                  hue_order=['SOFT', 'MEDIUM', 'HARD','INTERMEDIATE','WET'],
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

    styled = df2.style.apply(style_row, axis=1).set_table_styles(
        [{'selector': 'th', 'props': [('background-color', '#222'), ('color', '#fff')]}]
    ).set_properties(**{'border': '1px solid #999', 'padding': '6px'})
    return styled
    

def showracedetails(year, race_name, session_name):
    session = load_session_cached(year, race_name, session_name)

    fig1=showraceresults(session)
    fig2=tyreStrategies(session)
    fig3=driverlaptimes(session)
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

def getSpeedTraceFor(session, driver1, driver2):
    driver1_lap = session.laps.pick_drivers(driver1).pick_fastest()
    driver2_lap = session.laps.pick_drivers(driver2).pick_fastest()
    driver1_tel = driver1_lap.get_car_data().add_distance()
    driver2_tel = driver2_lap.get_car_data().add_distance()
    circuit_info = session.get_circuit_info()

    d1_color = fpl.get_team_color(driver1_lap['Team'], session=session)
    d2_color = fpl.get_team_color(driver2_lap['Team'], session=session)
    d2_linestyle = '--' if d1_color == d2_color else '-'

    fig, ax = plt.subplots()
    ax.plot(driver1_tel['Distance'], driver1_tel['Speed'], color=d1_color, label=driver1)
    ax.plot(driver2_tel['Distance'], driver2_tel['Speed'], color=d2_color, label=driver2, linestyle=d2_linestyle)

    # Draw vertical dotted lines at each corner that range from slightly below the
    # minimum speed to slightly above the maximum speed.
    v_min = driver1_tel['Speed'].min()
    v_max = driver1_tel['Speed'].max()
    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20,
            linestyles='dotted', colors='grey')

    # Plot the corner number just below each vertical line.
    # For corners that are very close together, the text may overlap. A more
    # complicated approach would be necessary to reliably prevent this.
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(corner['Distance'], v_min-30, txt,
                va='center_baseline', ha='center', size='small')

    ax.set_xlabel('Distance in m')
    ax.set_ylabel('Speed in km/h')

    ax.legend()
    plt.suptitle(f"Fastest Lap Comparison \n "
                f"{session.event['EventName']} {session.event.year} {session.name}")
    return fig

def getdriverstandings(year, round):
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=year, round=round)
    return standings.content[0]

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

def driverComparison(year, selected_race, selected_session, selected_driver1, selected_driver2):
    fpl.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
    session = load_session_cached(year, selected_race, selected_session)
    fig1 = getSpeedTraceFor(session, selected_driver1, selected_driver2)
    fig2 = showqualifyingdeltas(session, drv_list=[selected_driver1,selected_driver2])
    styled = showSectorTimesComparison(session, selected_driver1, selected_driver2)
    st.pyplot(fig2)
    st.pyplot(fig1)
    html = styled.to_html()
    components.html(html, height=300, scrolling=True)

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
        sprint = ergast.get_sprint_results(season=2022, round=rnd + 1)
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
    results.drop(columns='total_points', inplace=True)

    # Use race name, instead of round no., as column names
    results.columns = races

    fig = px.imshow(
        results,
        text_auto=True,
        aspect='auto',  # Automatically adjust the aspect ratio
        color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                                [0.25, 'rgb(107, 174, 214)'],
                                [0.5,  'rgb(33,  113, 181)'],
                                [0.75, 'rgb(8,   81,  156)'],
                                [1,    'rgb(8,   48,  107)']],
        labels={'x': 'Race',
                'y': 'Driver',
                'color': 'Points'}       # Change hover texts
    )
    fig.update_xaxes(title_text='')      # Remove axis titles
    fig.update_yaxes(title_text='')
    fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                    showline=False,
                    tickson='boundaries')              # Show horizontal grid only
    fig.update_xaxes(showgrid=False, showline=False)    # And remove vertical grid
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
    fig.update_layout(coloraxis_showscale=False)        # Remove legend
    fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # Remove border margins

    return fig


st.title("F1 Dashboard - FastF1")
st.sidebar.header("F1 Controls")

year = st.sidebar.slider("Select Year", min_value=1985, max_value=2026, value=2026)
schedule = getschedule(year)

#if st.sidebar.button("Driver Standings"):
with st.expander("Driver Standings", expanded=True):
    # Get the current drivers standings
    fig = showdriverstanding(year, round)
    st.plotly_chart(fig, use_container_width=True)

race_names = schedule['EventName'].tolist()
selected_race = st.sidebar.selectbox("Select Race", race_names)
st.write(f"You selected: {year} - {selected_race}")
race_info = schedule[schedule['EventName'] == selected_race].iloc[0]
round = int(race_info['RoundNumber'])
st.subheader("Selection Details")
st.write(f"Year: {year}")
st.write(f"Country: {race_info['Country']}")
st.write(f"Date: {str(race_info['EventDate'])}")
st.write(f"Round: {round}")
st.write(f"Race: {selected_race}")

if st.sidebar.button("Show Track"):
    with st.expander("Track Map", expanded=True):
        session = load_session_cached(year, selected_race, "FP1")
        fig = drawtrackfor(session)
        st.pyplot(fig)

sessions = []
print(schedule.columns)
for col in schedule.columns:
    if col.startswith("Session") and not col.endswith("Date") and not col.endswith("DateUtc"):
        val=race_info[col]
        if pd.notna(val):
            sessions.append(val)
selected_session = st.sidebar.selectbox("Select Session", sessions)

driver_codes = get_event_driver_abbreviations(year, selected_race, selected_session)
selected_driver1 = st.sidebar.selectbox("Select Driver", driver_codes)
selected_driver2 = st.sidebar.selectbox("Select Comparison Driver", driver_codes)


if st.sidebar.button("Compare Drivers"):
    with st.expander("Driver Comparison", expanded=True):
        driverComparison(year, selected_race, selected_session, selected_driver1, selected_driver2)

if st.sidebar.button("Show Session Details"):
    with st.expander("Race Details", expanded=True):
        showracedetails(year, selected_race, selected_session)

if st.sidebar.button("Who Can Win - Top 10"):
    with st.expander("Season Details", expanded=True):
        # Get the current drivers standings
        driver_standings = getdriverstandings(year, round)
        points = calculatemaxpointsforremainingseason(year, round)
        calculatewhocanwin(driver_standings, points)


