import streamlit as st
import pandas as pd
import numpy as np
import pickle, shap, os, json, urllib.request, warnings
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# main page configuration
st.set_page_config(
    page_title="DublinBikes Stock Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# injects custom css into the dashboard with google fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp, .main { background: #0e1117; }
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #252d3d;
}
#MainMenu, footer, header { visibility: hidden; }

/* Big risk box */
.risk-box {
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 16px;
}
.risk-safe    { background: #0d2a1a; border: 2px solid #1e6b3a; }
.risk-warning { background: #2a2000; border: 2px solid #9d7a00; }
.risk-danger  { background: #2a0a0a; border: 2px solid #8b1a1a; }

.risk-pct {
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.risk-msg {
    font-size: 1.1rem;
    font-weight: 500;
    margin-top: 10px;
}

/* Stat cards */
.stat-card {
    background: #1a2035;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    border: 1px solid #252d3d;
}
.stat-num { font-size: 2.4rem; font-weight: 700; color: #fff; }
.stat-lbl { font-size: 0.8rem; color: #7a8aab; margin-top: 4px;
            text-transform: uppercase; letter-spacing: 0.5px; }

/* Station card on map summary */
.summary-card {
    background: #1a2035;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #252d3d;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Alert pill */
.pill-safe    { background:#1e6b3a; color:#fff; padding:5px 14px;
                border-radius:30px; font-size:0.85rem; font-weight:600; }
.pill-warning { background:#9d7a00; color:#fff; padding:5px 14px;
                border-radius:30px; font-size:0.85rem; font-weight:600; }
.pill-danger  { background:#8b1a1a; color:#fff; padding:5px 14px;
                border-radius:30px; font-size:0.85rem; font-weight:600; }

.mode-pill-live { background:#1e6b3a; color:#fff; padding:3px 10px;
                  border-radius:20px; font-size:0.75rem; font-weight:700; }
.mode-pill-hist { background:#1a3060; color:#fff; padding:3px 10px;
                  border-radius:20px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE          = r"D:\Griffith College Docs year 2\01 Dissertation"
FEATURES_PATH = os.path.join(BASE, "merged-datasets", "features_dataset.parquet")
MODELS_DIR    = os.path.join(BASE, "Models")
XGB_PATH      = os.path.join(MODELS_DIR, "xgboost.pkl")

API_KEY  = os.environ.get("JCDECAUX_KEY", "")
JCAPI    = f"https://api.jcdecaux.com/vls/v1/stations?contract=dublin&apiKey={API_KEY}"
METAPI   = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=53.3617&longitude=-6.3206"
    "&current=temperature_2m,relative_humidity_2m,precipitation,"
    "surface_pressure,dew_point_2m"
    "&timezone=Europe%2FDublin"
)

FEATURE_COLS = [
    'bikes_min', 'docks_min', 'capacity', 'reading_count',
    'occupancy_ratio', 'docks_occupancy_ratio',
    'bikes_lag_1h', 'bikes_lag_2h', 'bikes_lag_3h',
    'bikes_lag_4h', 'bikes_lag_5h',
    'bikes_lag_6h', 'bikes_lag_12h', 'bikes_lag_24h',
    'docks_lag_1h', 'docks_lag_2h', 'docks_lag_3h',
    'rolling_mean_3h', 'rolling_mean_6h', 'rolling_std_3h',
    'bikes_change_1h', 'bikes_change_3h', 'occupancy_lag_1h',
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_public_holiday', 'is_working_day',
    'rain', 'temp', 'rhum', 'msl', 'wetb', 'dewpt', 'vappr',
    'is_imputed'
]

# ── LOAD ───────────────────────────────────────────────────────────────────────
# Loads the .pkl model from the folder and saves within memory
@st.cache_resource
def load_model():
    with open(XGB_PATH, "rb") as f:
        return pickle.load(f)

# Creates the shap tree explainer from loaded model
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

# loads column names in the parquet order and filters the dates from 2024 onwards.
@st.cache_data
def load_data():
    cols = ["datetime_hour","station_id","name","lat","lon"] + FEATURE_COLS + ["will_stockout_1h"]
    df = pd.read_parquet(FEATURES_PATH, columns=cols)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    return df[df["datetime_hour"] >= "2024-01-01"].copy().reset_index(drop=True) # This date here can be changed to alter dataset size that should be loaded

# Builts the lag lookup table for LIVE mode for a historical idea of the data
@st.cache_data
def load_lookup():
    df = pd.read_parquet(FEATURES_PATH,
                         columns=["datetime_hour","station_id","bikes_min","docks_min"])
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df = df[df["datetime_hour"] >= "2023-01-01"].sort_values(["station_id","datetime_hour"])
    return df.set_index(["station_id","datetime_hour"])

# Computes the fallback values for lag features in Live Mode such that end of dataset in live mode deos'nt affect that integrity
@st.cache_data
def load_medians():
    df = pd.read_parquet(FEATURES_PATH,
                         columns=["datetime_hour","station_id","bikes_min","docks_min"])
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    m = df.groupby(["station_id","hour"]).agg(
        bmed=("bikes_min","median"), dmed=("docks_min","median") # Bikes and Docks Median
    ).reset_index()
    return m.set_index(["station_id","hour"])

# ── HELPERS ────────────────────────────────────────────────────────────────────
# COnverts the raw flat probability into five display elements
# returns a tuple of percentage number, human readable message, css class for coloured box, pill badge and pill text 
def risk_info(prob):
    pct = prob * 100
    if prob >= 0.90:
        return pct, "🚨 Urgent: likely to run out soon", "risk-danger", "pill-danger", "🚨 URGENT"
    elif prob >= 0.60:
        return pct, "⚠️ Caution: running low", "risk-warning", "pill-warning", "⚠️ CAUTION"
    else:
        return pct, "✅ Safe: plenty of bikes", "risk-safe", "pill-safe", "✅ SAFE"

# Creates the circular gauge dial using plotly for more user friendly appearance
def gauge(prob):
    pct = prob * 100

    # calls risk_info and unpakcs the fith element (pill text) and discard others
    _, _, _, _, pill = risk_info(prob)

    # colors based on risk percentage
    colour = "#ef4444" if prob>=0.90 else "#eab308" if prob>=0.60 else "#22c55e"

    # creates the gauge icon with percentage number and adds style into it
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix":"%","font":{"size":44,"color":colour,"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#4a5568",
                    "tickfont":{"color":"#4a5568","size":11}},
            "bar":{"color":colour,"thickness":0.25},
            "bgcolor":"#1a2035","bordercolor":"#252d3d",
            "steps":[
                {"range":[0,60], "color":"#0d2a1a"},
                {"range":[60,90],"color":"#2a2000"},
                {"range":[90,100],"color":"#2a0a0a"},
            ],
        }
    ))

    # sets background of chart same as the rest of the background
    fig.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                      height=240,margin=dict(l=20,r=20,t=10,b=10))
    return fig

# Creates the shap image, changes their column names.
def shap_fig(sv, fv, base): # Take in 37 SHAP values for one prediction, 37 actual feature values for same prediction and model's avg predicted probability across training data
    # Friendly names for SHAP waterfall
    name_map = {
        'bikes_min':'Bikes right now','docks_occupancy_ratio':'Docks fullness',
        'occupancy_ratio':'Station fullness','hour':'Hour of day',
        'bikes_lag_1h':'Bikes 1h ago','bikes_change_1h':'Bike trend (1h)',
        'docks_min':'Free docks','reading_count':'Data quality',
        'bikes_lag_12h':'Bikes 12h ago','bikes_lag_24h':'Bikes 24h ago',
        'temp':'Temperature','rain':'Rainfall',
        'is_working_day':'Working day','is_weekend':'Weekend',
    }
    fnames = [name_map.get(c, c.replace("_"," ").title()) for c in FEATURE_COLS] # looks up the friendly name from dictionary and if not found, falls back to replacing them with spaces and capitalisation
    exp = shap.Explanation(values=sv, base_values=base, data=fv, feature_names=fnames)
    fig, ax = plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # draws the waterfall chart
    shap.waterfall_plot(exp, max_display=10, show=False)
    ax.set_title("What drove this prediction?",color="#cbd5e1",fontsize=12,pad=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#3d3725")
    ax.tick_params(colors="#7a8aab",labelsize=9)
    ax.xaxis.label.set_color("#7a8aab")
    plt.tight_layout()
    return fig

# Builds the interactive Dublin map with score. Each circle's color and score are driven by risk_pct
def net_map(df_map):
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon",
        color="risk_pct", size="risk_pct", size_max=22,
        hover_name="name",
        hover_data={"risk_pct":":.0f","bikes":True,"alert":True,
                    "lat":False,"lon":False},
        color_continuous_scale=[[0,"#22c55e"],[0.6,"#eab308"],[0.9,"#ef4444"],[1,"#991b1b"]],
        range_color=[0,100],
        mapbox_style="open-street-map", # Can even use: "carto-darkmatter"; if the open-street-map fails to load (Shows API KEY REQUIRED)
        zoom=12.5, center={"lat":53.344,"lon":-6.267},
        labels={"risk_pct":"Risk %","bikes":"Bikes Now","alert":"Status"}
    )
    fig.update_layout(
        paper_bgcolor="#0e1117", height=500,
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Risk %",font=dict(color="#7a8aab")),
            tickfont=dict(color="#7a8aab")
        )
    )
    return fig

# ── LIVE API ───────────────────────────────────────────────────────────────────
# creates a GET request to JCDecaux api to recieve live data
def fetch_bikes():
    if not API_KEY:
        return None, ("No JCDecaux API key found. Get a free key at "
                      "https://developer.jcdecaux.com and set it as an "
                      "environment variable named JCDECAUX_KEY.")
    
    try:

        # Creates an HTTP req with user-agent header with a timeout of 10s
        req = urllib.request.Request(JCAPI,
              headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=10) as r:
            raw = json.loads(r.read()) # Read the response and parse the JSON into Python list
        rows=[]
        for s in raw:
            # skips closed stations 
            if s.get("status")!="OPEN": continue
            b=s.get("available_bikes",0); d=s.get("available_bike_stands",0)
            # skips stations with zero bikes and zero docks
            if b==0 and d==0: continue

            # build dictionary for each valid station with standard field names
            rows.append({"station_id":s["number"],"name":s["name"], "bikes_live":b,"docks_live":d, "capacity":s.get("bike_stands",1), "lat":s["position"]["lat"],"lon":s["position"]["lng"]})
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, str(e)

# Fetches the current weather from Open-Meteo
def fetch_weather(df_hist):
    try:
        req = urllib.request.Request(METAPI,
              headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=10) as r:
            data = json.loads(r.read())
        cur = data.get("current",{})
        t = float(cur.get("temperature_2m",18) or 18)
        rh= float(cur.get("relative_humidity_2m",70) or 70)
        rn= float(cur.get("precipitation",0) or 0)
        ms= float(cur.get("surface_pressure",1013) or 1013)
        dp= float(cur.get("dew_point_2m",8) or 8)

        # Calculates wetb temperature using Stull approximation (lowest temperature that can be reached by water evaporation)
        wetb = (t*np.arctan(0.151977*(rh+8.313659)**0.5)
                +np.arctan(t+rh)-np.arctan(rh-1.676331)
                +0.00391838*rh**1.5*np.arctan(0.023101*rh)-4.686035)

        # Magnus formulae for vapour pressure
        vappr = (rh/100)*6.1078*np.exp(17.27*t/(t+237.3))
        return {"rain":rn,"temp":t,"rhum":rh,"msl":ms,"wetb":round(wetb,1),
                "dewpt":dp,"vappr":round(vappr,1)}, None
    except Exception as e:
        # Fallback to most recent dataset weather if API Fails
        row = df_hist.sort_values("datetime_hour").tail(1).iloc[0]
        wx = {c:float(row[c]) for c in ["rain","temp","rhum","msl","wetb","dewpt","vappr"]}
        return wx, str(e)

# Looks up one lag value for one station at one time offset.
#every lag value is actually hisotorical median
def get_lag(lookup, medians, sid, now, h, col="bikes_min"):
    try:
        return float(lookup.loc[(sid, now-pd.Timedelta(hours=h)), col]) # subtract h hours from current timestamp to get the target
    except:
        past_h = (now-pd.Timedelta(hours=h)).hour #looks up time for that station (if now is 15:00 and h=1; looks for 15-1=14 or 2PM)
        try:
            row = medians.loc[(sid, past_h)] # Looks up median for this station at 14:00 across entire 2019 to 2026
            return float(row["bmed"] if col=="bikes_min" else row["dmed"])
        except:
            return np.nan

# this creates the complete 37-elements feature vector for a single live station in the exact same order
# firstly, it extracts 3 current state values from the LIVE API row
def build_features(live_row, lookup, medians, wx, now):
    sid  = live_row["station_id"]
    bmin = float(live_row["bikes_live"])
    dmin = float(live_row["docks_live"])
    cap  = max(float(live_row["capacity"]),1)

    # calls get_lag() 11 times to get all the lag values. Each cell returns a real historical value or median fallback
    def lb(h): return get_lag(lookup,medians,sid,now,h,"bikes_min")
    def ld(h): return get_lag(lookup,medians,sid,now,h,"docks_min")
    bl1,bl2,bl3 = lb(1),lb(2),lb(3)
    bl4,bl5,bl6 = lb(4),lb(5),lb(6)
    bl12,bl24   = lb(12),lb(24)
    dl1,dl2,dl3 = ld(1),ld(2),ld(3)

    # if it returned Nan, replace the current live bike count as last resort
    def f(v): return bmin if (v is None or np.isnan(v)) else v

    # computes mean, sttandrad deviation and ignores nan values
    rm3 = np.nanmean([bl1,bl2,bl3])
    rm6 = np.nanmean([bl1,bl2,bl3,bl4,bl5,bl6])
    rs3 = float(np.nanstd([bl1,bl2,bl3]))
    ch1 = bmin-f(bl1); ch3 = bmin-f(bl3) # Current bikes 1h ago and 3h ago

    # build is_weekend, is_public_holiday and is_working_day flags
    iw = int(now.dayofweek>=5)
    iph = 0
    iwd = int(not iw and not iph)

    # Assembles all the 37 values into numpy array in exact order of the original dataset
    return np.array([
        bmin,dmin,cap,6.0,
        bmin/cap, dmin/cap,
        f(bl1),f(bl2),f(bl3),f(bl4),f(bl5),
        f(bl6),f(bl12),f(bl24),
        f(dl1),f(dl2),f(dl3),
        f(rm3),f(rm6),f(rs3),
        ch1,ch3,f(bl1)/cap,
        float(now.hour),float(now.dayofweek),float(now.month),
        float(iw),float(iph),float(iwd),
        wx["rain"],wx["temp"],wx["rhum"],wx["msl"],
        wx["wetb"],wx["dewpt"],wx["vappr"],
        0.0
    ], dtype="float32")

# FRONT-END
# Generates the frontend of the streamlit dashboard using all the methods that was built earlier
# All data is already cached and are just being call here
def main():
    model    = load_model()
    explainer= load_explainer(model)
    df       = load_data()
    lookup   = load_lookup()
    medians  = load_medians()

    # SIDEBAR
    with st.sidebar:
        st.markdown("## 🚲 DublinBikes Stock Predictor")
        st.markdown("---")
        live = st.toggle("🔴 Go LIVE (JCDecaux API)", value=False)

        # Historical Mode
        if not live:
            st.markdown("### 🔍 Browse History")
            stations = sorted(df["name"].unique()) # Gets all unique station names sorted alphabetically

            # Creates station dropdown, date calendar dropdown and hour slider
            sel = st.selectbox("Pick a station", stations)
            mind = df["datetime_hour"].min().date()
            maxd = df["datetime_hour"].max().date()
            sel_date = st.date_input("Date", value=maxd,
                                     min_value=mind, max_value=maxd)
            sel_hour = st.slider("Hour", 0, 23, 9)

        # LIVE Mode
        # Deletes cached data when refreshed and fetches new data causing load_data(), load_lookup and load_medians to reload
        else:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.cache_data.clear()

        st.markdown("---")
        st.markdown("""
        **How to read alerts:**
        - 🟢 **SAFE**: below 60%
        - 🟡 **CAUTION**: 60 to 89%
        - 🔴 **URGENT**: 90% and above

        """)

    # HEADER
    badge = '<span class="mode-pill-live">🔴 LIVE</span>' if live \
            else '<span class="mode-pill-hist">📂 HISTORY</span>'
    st.markdown(f"# 🚲 DublinBikes Stock Predictor &nbsp; {badge}", unsafe_allow_html=True)
    st.markdown("**Predicts whether a Dublin Bikes station will run out of bikes in the next hour.**")
    st.markdown("---")

    # LIVE MODE
    if live:
        # Shows a rotating circle symbolising loading screen
        with st.spinner("Getting live bike data from JCDecaux..."):
            live_df, err = fetch_bikes()

        if err or live_df is None:
            st.warning(f"⚠️ JCDecaux API unavailable ({err}). Showing most recent historical data instead.")
            # Fall back to the most recent available hour in the features dataset
            latest_dt = df["datetime_hour"].max()
            fallback = df[df["datetime_hour"] == latest_dt].copy()
            live_df = pd.DataFrame({
                "station_id": fallback["station_id"].values,
                "name":       fallback["name"].values,
                "bikes_live": fallback["bikes_min"].values,
                "docks_live": fallback["docks_min"].values,
                "capacity":   fallback["capacity"].values,
                "lat":        fallback["lat"].values,
                "lon":        fallback["lon"].values,
            })
            now = latest_dt
            st.info(f"📅 Showing data from: {latest_dt.strftime('%d %b %Y %H:%M')} (last available in dataset)")

        else:
            now = pd.Timestamp.now().floor("h")

        with st.spinner("Getting current weather..."):
            wx, wx_err = fetch_weather(df)

        # if weather fails, fallback to last known weather values on May 2026
        if wx_err:
            st.warning("Weather API unavailable, using last known values.")
        else:
            st.success(f"🌤️ Dublin right now: **{wx['temp']}°C** · "
                       f"Rain: {wx['rain']}mm · Humidity: {wx['rhum']:.0f}%")

        st.write(f"📅 Predictions for **{now.strftime('%A %d %B %Y : %H:%M')}** "
                 f"| {len(live_df)} active stations")

        # loops through each station row, converts it into a dictionary and returns a 37-element numpy array
        with st.spinner("Running predictions for all stations..."):
            feats = np.vstack([build_features(r.to_dict(),lookup,medians,wx,now)
                               for _,r in live_df.iterrows()])
            probs = model.predict_proba(feats)[:,1]

        # adds two new columns to live dataframe
        live_df["risk_pct"] = probs*100
        live_df["bikes"]    = live_df["bikes_live"].fillna(0).astype(int)

        # this applies the threshold logic to assign text label used in the map's hover tooltip
        live_df["alert"]    = live_df["risk_pct"].apply(
            lambda p:"🔴 URGENT" if p>=90 else "🟡 CAUTION" if p>=60 else "🟢 SAFE")

        # counts stations in each alert category
        n_urg = (live_df["risk_pct"]>=90).sum()
        n_cau = ((live_df["risk_pct"]>=60)&(live_df["risk_pct"]<90)).sum()
        n_saf = (live_df["risk_pct"]<60).sum()

        # Creates the 4 summary cards that include urgent, caution, safe and total stations numbers
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-num" style="color:#ef4444">{n_urg}</div>
                <div class="stat-lbl">🔴 Urgent</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-num" style="color:#eab308">{n_cau}</div>
                <div class="stat-lbl">🟡 Caution</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-num" style="color:#22c55e">{n_saf}</div>
                <div class="stat-lbl">🟢 Safe</div></div>""",
                unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-num">{len(live_df)}</div>
                <div class="stat-lbl">Total Stations</div></div>""",
                unsafe_allow_html=True)

        # Creates 3 tabs: map, station details and urgent stations
        st.markdown("")
        tab_map, tab_detail, tab_urgent = st.tabs(
            ["🗺️ Map", "🔍 Station Detail", "🚨 Urgent Stations"])

        # generates the map tab
        with tab_map:
            st.markdown("#### Dublin Bikes Live Risk Map")
            st.caption("Green = safe · Yellow = caution · Red = urgent")
            st.plotly_chart(net_map(live_df), use_container_width=True)

        # generates the station detail tab
        with tab_detail:

            # station selection dropdown menu
            sel_live = st.selectbox("Choose a station to inspect", sorted(live_df["name"].tolist()))
            row_l = live_df[live_df["name"]==sel_live].iloc[0]
            pos_l = list(live_df["name"]).index(sel_live)
            fv_l  = feats[pos_l]
            prob_l= float(row_l["risk_pct"])/100
            pct,msg,box_cls,_,pill = risk_info(prob_l)

            col_g, col_info = st.columns([1,2])

            # Generates the gauge and pproabability percentage
            with col_g:
                st.plotly_chart(gauge(prob_l), use_container_width=True)
                st.markdown(f"<div style='text-align:center;font-size:1.1rem;"
                            f"font-weight:700;color:{'#ef4444' if prob_l>=0.90 else '#eab308' if prob_l>=0.60 else '#22c55e'}'>"
                            f"{pill}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;color:#94a3b8;"
                            f"margin-top:8px'>{msg}</div>", unsafe_allow_html=True)

            with col_info:

                # displays station name and a markdown
                st.markdown(f"### {sel_live.title()}")
                st.markdown(f"**Prediction: next 1 hour from now**")
                st.markdown("")

                # Generates 3 summary cards for that station including bikes right now, free docks and total capacity of the station right now
                m1,m2,m3 = st.columns(3)
                bikes_now = int(row_l["bikes_live"])
                docks_now = int(row_l["docks_live"])
                cap_now   = int(row_l["capacity"])

                # Color for bikes_now card which changes depending on total bikes present
                bike_col = "#ef4444" if bikes_now==0 else \
                           "#eab308" if bikes_now<=3 else "#22c55e"

                # Design for the three summary cards in the station
                with m1:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-num" style="color:{bike_col}">{bikes_now}</div>
                        <div class="stat-lbl">🚲 Bikes Now</div></div>""",
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-num">{docks_now}</div>
                        <div class="stat-lbl">🅿️ Free Docks</div></div>""",
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-num">{cap_now}</div>
                        <div class="stat-lbl">📊 Capacity</div></div>""",
                        unsafe_allow_html=True)

                st.markdown("")
                st.markdown(f"🌤️ Weather: **{wx['temp']}°C** · "
                            f"Rain: {wx['rain']}mm · Humidity: {wx['rhum']:.0f}%")

            # SHAP image generation which plots the prediction reasons
            st.markdown("---")
            st.markdown("#### Why did the model predict this?")
            st.caption("The chart below shows which factors had the biggest influence on this prediction. "
                       "Bars going right = increased risk. Bars going left = decreased risk.")
            with st.spinner("Explaining..."):
                sv = explainer.shap_values(fv_l.reshape(1,-1))[0]
            fig_wf = shap_fig(sv, fv_l, explainer.expected_value)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

        # urgent tab
        with tab_urgent:
            urg_df = live_df[live_df["risk_pct"]>=90].sort_values(
                "risk_pct", ascending=False)

            # if no stations at urgent attention, else show all stations with urgent attention
            if len(urg_df)==0:
                st.success("No stations at urgent risk right now! 🎉")
            else:
                st.error(f"**{len(urg_df)} stations need attention right now**")
                for _, row in urg_df.iterrows():
                    bcol = "#ef4444" if row["bikes_live"]==0 else "#eab308"
                    st.markdown(f"""
                    <div style="background:#1a2035;border-radius:12px;padding:16px 20px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                    <div style="font-weight:600;font-size:1rem;color:#e2e8f0">{row['name'].title()}</div>
                    <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px"><span style="color:{bcol};font-weight:600">{int(row['bikes_live'])} bikes</span> available now</div>
                    </div>   
                    <span class="pill-danger">{row['risk_pct']:.0f}%</span>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)


    # HISTORICAL MODE
    # Constructs a pandas timestamp from sidebar date picker and hour slider and searches for that exact row
    # If not found, searches for the nearest timestamp
    else:
        target_dt = pd.Timestamp(year=sel_date.year, month=sel_date.month,
                                  day=sel_date.day, hour=sel_hour)

        row_df = df[(df["name"]==sel) & (df["datetime_hour"]==target_dt)]
        if row_df.empty:
            st_df = df[df["name"]==sel]
            if st_df.empty:
                st.error("No data found for this station."); return
            row_df = st_df.iloc[
                [(st_df["datetime_hour"]-target_dt).abs().argsort().iloc[0:1]]]
            st.info(f"Showing nearest available time: "
                    f"{row_df.iloc[0]['datetime_hour'].strftime('%d %b %Y %H:%M')}")

        row = row_df.iloc[0]

        # selects the 37 feature values in correct order, converts to a numpy array and makes it 1, 37. [0,1] gets the stock-out probability from output
        X   = row[FEATURE_COLS].to_numpy(dtype="float32").reshape(1,-1)
        prob= float(model.predict_proba(X)[0,1])
        pct,msg,box_cls,_,pill = risk_info(prob)

        # Top section: gauge + info
        col_g, col_info = st.columns([1,2])

        # Generates the gauge + probability
        with col_g:
            st.markdown(f"### {sel.title()}")
            st.caption(row["datetime_hour"].strftime("%A %d %B %Y · %H:%M"))
            st.plotly_chart(gauge(prob), use_container_width=True)
            colour = "#ef4444" if prob>=0.90 else "#eab308" if prob>=0.60 else "#22c55e"
            st.markdown(f"<div style='text-align:center;font-size:1.1rem;"
                        f"font-weight:700;color:{colour}'>{pill}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;color:#94a3b8;"
                        f"margin-top:6px'>{msg}</div>", unsafe_allow_html=True)

        # Generates the 4 summary card: minimum bikes, free docks, capacity and bikes change trend in last 1 hour
        with col_info:
            st.markdown(f"### What is happening?")
            bikes = int(row["bikes_min"])
            docks = int(row["docks_min"])
            cap   = int(row["capacity"])
            bcol  = "#ef4444" if bikes==0 else "#eab308" if bikes<=3 else "#22c55e"

            m1,m2,m3,m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-num" style="color:{bcol}">{bikes}</div>
                    <div class="stat-lbl">🚲 Bikes</div></div>""",
                    unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-num">{docks}</div>
                    <div class="stat-lbl">🅿️ Free Docks</div></div>""",
                    unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-num">{cap}</div>
                    <div class="stat-lbl">📊 Capacity</div></div>""",
                    unsafe_allow_html=True)
            with m4:
                chg = row["bikes_change_1h"]
                arr = "📉" if chg<0 else "📈" if chg>0 else "➡️"
                chg_col = "#ef4444" if chg<0 else "#22c55e" if chg>0 else "#94a3b8"
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-num" style="color:{chg_col}">{arr}{abs(chg):.0f}</div>
                    <div class="stat-lbl">Trend (1h)</div></div>""",
                    unsafe_allow_html=True)

            # weather markdown showing current weather (at that timestamp)
            st.markdown("")
            st.markdown(f"🌤️ Weather at this time: **{row['temp']:.1f}°C** · "
                        f"Rain: {row['rain']:.1f}mm · Humidity: {row['rhum']:.0f}%")

            # Simple plain-language summary
            st.markdown("")
            if prob >= 0.90:
                st.error(f"🚨 **High chance of running out of bikes.** "
                         f"The model predicts a {pct:.0f}% probability that "
                         f"this station will have zero bikes within the next hour.")
            elif prob >= 0.60:
                st.warning(f"⚠️ **Running low.** "
                           f"There is a {pct:.0f}% chance this station "
                           f"could run out of bikes in the next hour.")
            else:
                st.success(f"✅ **Looking good.** "
                           f"Only a {pct:.0f}% chance of running out. "
                           f"Plenty of bikes available.")

        st.markdown("---")

        # Two columns: SHAP + history
        col_shap, col_hist = st.columns([3,2])

        # shows the SHAP diagram for model prediction
        with col_shap:
            st.markdown("#### Why did the model predict this?")
            st.caption("Bars going right = things that increased the risk. "
                       "Bars going left = things that lowered the risk.")
            with st.spinner("Explaining..."):
                sv = explainer.shap_values(X)[0]
            fig_wf = shap_fig(sv, X[0], explainer.expected_value)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

        # generates a graph for the probabilities of the last 24 hours in this station
        with col_hist:
            st.markdown("#### Last 24 hours at this station")
            hist = df[(df["name"]==sel) &
                      (df["datetime_hour"]>=target_dt-timedelta(hours=23)) &
                      (df["datetime_hour"]<=target_dt)].sort_values("datetime_hour")
            if not hist.empty:
                Xh = hist[FEATURE_COLS].to_numpy(dtype="float32")
                ph = model.predict_proba(Xh)[:,1]

                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=hist["datetime_hour"], y=ph*100,
                    mode="lines+markers",
                    line=dict(color="#7eb3ff",width=2),
                    fill="tozeroy",fillcolor="rgba(126,179,255,0.1)"
                ))
                fig_r.add_hline(y=60,line_dash="dot",line_color="#eab308",opacity=0.7,
                                annotation_text="Caution 60%")
                fig_r.add_hline(y=90,line_dash="dot",line_color="#ef4444",opacity=0.7,
                                annotation_text="Urgent 90%")
                fig_r.update_layout(
                    paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=190,
                    margin=dict(l=10,r=10,t=10,b=10),title="Predicted Risk %",
                    title_font_color="#94a3b8",title_font_size=11,
                    xaxis=dict(tickfont=dict(color="#4a5568"),gridcolor="#1e2535"),
                    yaxis=dict(tickfont=dict(color="#4a5568"),gridcolor="#1e2535",
                               range=[0,105]),
                    showlegend=False,font=dict(color="#e2e8f0")
                )
                st.plotly_chart(fig_r, use_container_width=True)

                # Generates the second bar graph showing bikes available
                fig_b = go.Figure(go.Bar(
                    x=hist["datetime_hour"], y=hist["bikes_min"],
                    marker_color=["#ef4444" if b==0 else
                                  "#eab308" if b<=3 else "#22c55e"
                                  for b in hist["bikes_min"]]
                ))
                fig_b.update_layout(
                    paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                    height=160,margin=dict(l=10,r=10,t=10,b=10),
                    title="Bikes Available",title_font_color="#94a3b8",
                    title_font_size=11,
                    xaxis=dict(tickfont=dict(color="#4a5568"),gridcolor="#1e2535"),
                    yaxis=dict(tickfont=dict(color="#4a5568"),gridcolor="#1e2535"),
                    showlegend=False,font=dict(color="#e2e8f0")
                )
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.info("No historical data for this time window.")

        st.markdown("---")

        # Network map for the selected time
        # Generates a map that displays all station's conditions with risk percentage and bikes available now
        st.markdown("#### All stations at this time")
        map_df = df[df["datetime_hour"]==row["datetime_hour"]].copy()
        if not map_df.empty:
            Xm = map_df[FEATURE_COLS].to_numpy(dtype="float32")
            map_df["risk_pct"] = model.predict_proba(Xm)[:,1]*100
            map_df["bikes"]    = map_df["bikes_min"]
            map_df["alert"]    = map_df["risk_pct"].apply(
                lambda p:"🔴 URGENT" if p>=90 else "🟡 CAUTION" if p>=60 else "🟢 SAFE")
            st.plotly_chart(net_map(map_df), use_container_width=True)

# starts the application
if __name__ == "__main__":
    main()