import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import traceback
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# =========================================
# CONFIG (edit these only)
# =========================================
TZ=ZoneInfo("Asia/Kuala_Lumpur")
REFRESH_MS=3000
ACTIVE_SEC=120
WINDOW_SEC=30
MAX_EVENTS_WINDOW=10
ALERT_LOOKBACK_MIN=30
HEATMAP_LOOKBACK_HR=2

# IMPORTANT: Use the PUBLISHED CSV link (NOT the /edit link)
SHEET_CSV_URL="https://docs.google.com/spreadsheets/d/e/2PACX-1vSxyGtEAyftAfaY3M3H_sMvnA6oYcTsVjxMLVznP7SXvGA4rTXfrvzESYgSND7Z6o9qTrD-y0QRyvPo/pub?gid=0&single=true&output=csv"

st.set_page_config(page_title="Stray Dog Control System",layout="wide")
st_autorefresh(interval=REFRESH_MS,key="data_refresh")

# =========================================
# LIGHT UI STYLE (CSS)
# =========================================
st.markdown("""
<style>
html, body, [class*="css"] {background:#f6f7fb !important;}
.block-container {padding-top:1.2rem;}
.header-bar {background:#ffffff;border:1px solid #e8eaf0;border-radius:16px;padding:16px 18px;box-shadow:0 2px 10px rgba(16,24,40,0.04);}
.title {font-size:28px;font-weight:800;color:#0f172a;margin:0;}
.subtitle {font-size:13px;color:#64748b;margin-top:4px;}
.clock {text-align:right;color:#0f172a;font-weight:700;}
.date {text-align:right;color:#64748b;font-size:12px;margin-top:2px;}
.kpi-card {background:#ffffff;border:1px solid #e8eaf0;border-radius:16px;padding:14px 14px;box-shadow:0 2px 10px rgba(16,24,40,0.04);}
.kpi-label {font-size:12px;color:#64748b;margin-bottom:6px;display:flex;gap:8px;align-items:center;}
.kpi-value {font-size:22px;font-weight:800;color:#0f172a;line-height:1;}
.kpi-sub {font-size:12px;color:#64748b;margin-top:6px;}
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid #e8eaf0;background:#f8fafc;color:#0f172a;}
.badge-high {background:#ffe4e6;border-color:#fecdd3;color:#9f1239;}
.badge-med {background:#fff7ed;border-color:#fed7aa;color:#9a3412;}
.badge-low {background:#ecfeff;border-color:#a5f3fc;color:#155e75;}
.panel {background:#ffffff;border:1px solid #e8eaf0;border-radius:16px;padding:14px 14px;box-shadow:0 2px 10px rgba(16,24,40,0.04);}
.panel-title {font-size:14px;font-weight:800;color:#0f172a;display:flex;align-items:center;gap:8px;margin-bottom:10px;}
.smallbox {background:#f8fafc;border:1px solid #e8eaf0;border-radius:12px;padding:10px;}
.alert-item {border-left:6px solid #e8eaf0;background:#ffffff;border:1px solid #e8eaf0;border-radius:14px;padding:10px 10px;margin-bottom:10px;}
.alert-time {font-size:12px;color:#64748b;margin-bottom:2px;}
.alert-title {font-size:13px;font-weight:800;color:#0f172a;display:flex;justify-content:space-between;align-items:center;gap:10px;}
.alert-meta {font-size:12px;color:#334155;margin-top:4px;}
.footer {color:#64748b;font-size:12px;margin-top:6px;}
</style>
""",unsafe_allow_html=True)

# =========================================
# FUZZY SEVERITY (membership functions)
# Inputs:
#   C: confidence (0..1)
#   N: dog_count (>=1)
#   R: event_rate (0..1) within WINDOW_SEC
# Output:
#   severity label + risk score (0..1)
# =========================================
def trapmf(x,a,b,c,d):
    x=float(x)
    if x<=a or x>=d:
        return 0.0
    if b<=x<=c:
        return 1.0
    if a<x<b:
        return (x-a)/(b-a) if b-a!=0 else 0.0
    return (d-x)/(d-c) if d-c!=0 else 0.0

def severity_fuzzy_basic(C,N,R):
    conf_high=trapmf(C,0.60,0.75,1.00,1.00)
    conf_med=trapmf(C,0.40,0.55,0.75,0.90)
    many=trapmf(min(N/3.0,1.0),0.40,0.60,1.00,1.00)
    freq_high=trapmf(R,0.40,0.60,1.00,1.00)
    r_high=max(min(conf_high,freq_high),many)
    r_med=max(min(conf_med,freq_high),min(conf_high,trapmf(R,0.20,0.35,0.60,0.80)))
    r_low=max(trapmf(C,0.00,0.00,0.35,0.55),trapmf(R,0.00,0.00,0.15,0.30))
    score=(r_low*0.2+r_med*0.6+r_high*0.9)/(r_low+r_med+r_high+1e-9)
    if score>=0.75:
        return "HIGH",score
    if score>=0.45:
        return "MED",score
    return "LOW",score

# =========================================
# DATA LOADING (robust against bad CSV rows)
# The published CSV sometimes contains messy rows (extra commas)
# This loader will:
# - try pd.read_csv normally
# - if fails: fetch raw text and use python engine + on_bad_lines="skip"
# =========================================
def load_data_from_sheet(url):
    try:
        df=pd.read_csv(url)
    except Exception:
        text=requests.get(url,timeout=20).text
        df=pd.read_csv(io.StringIO(text),engine="python",on_bad_lines="skip")

    df.columns=[c.strip() for c in df.columns]

    # Ensure expected columns exist
    expected=["timestamp","camera_id","location","class","confidence","dog_count","image_url"]
    for c in expected:
        if c not in df.columns:
            df[c]=None

    # Clean/convert types
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    df=df.dropna(subset=["timestamp"]).sort_values("timestamp",ascending=True)

    df["confidence"]=pd.to_numeric(df["confidence"],errors="coerce").fillna(0.0).clip(0,1)
    df["dog_count"]=pd.to_numeric(df["dog_count"],errors="coerce").fillna(1).astype(int)

    df["camera_id"]=df["camera_id"].fillna("unknown").astype(str)
    df["location"]=df["location"].fillna("unknown").astype(str)
    df["class"]=df["class"].fillna("dog").astype(str)
    df["image_url"]=df["image_url"].fillna("").astype(str)

    return df

# =========================================
# Add rolling event_rate + severity columns
# =========================================
def add_event_rate_and_severity(df):
    if df.empty:
        df["event_rate"]=0.0
        df["severity"]="LOW"
        df["risk_score"]=0.0
        return df

    df=df.copy().sort_values("timestamp",ascending=True)

    # event_rate: how many events in last WINDOW_SEC (per camera), normalized 0..1
    def _per_camera(g):
        g=g.sort_values("timestamp").copy()
        g=g.set_index("timestamp")
        cnt=g["confidence"].rolling(f"{WINDOW_SEC}s").count()
        g["event_rate"]=(cnt/MAX_EVENTS_WINDOW).clip(0,1)
        return g.reset_index()

    df=df.groupby("camera_id",dropna=False,group_keys=False).apply(_per_camera)

    # severity + risk score
    sev=df.apply(lambda r: pd.Series(severity_fuzzy_basic(r["confidence"],r["dog_count"],r["event_rate"])),axis=1)
    df["severity"]=sev.iloc[:,0]
    df["risk_score"]=sev.iloc[:,1]
    return df

def conf_label(c):
    if c>=0.80:
        return "High"
    if c>=0.55:
        return "Medium"
    return "Low"

def badge_html(level):
    if level=="HIGH":
        return "<span class='badge badge-high'>HIGH</span>"
    if level=="MED":
        return "<span class='badge badge-med'>MED</span>"
    return "<span class='badge badge-low'>LOW</span>"

def kpi_card(icon,label,value,sub):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{icon}<span>{label}</span></div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """

# =========================================
# Heatmap (no coordinates)
# Deterministic pseudo-position per camera+location
# =========================================
def stable_xy(key):
    h=abs(hash(key))%(10**8)
    x=(h%1000)/999.0
    y=((h//1000)%1000)/999.0
    return x,y

def render_heatmap(df_recent):
    grid=np.zeros((120,120),dtype=float)
    for _,r in df_recent.iterrows():
        k=f"{r.get('camera_id','')}-{r.get('location','')}"
        x,y=stable_xy(k)
        cx=int(x*(grid.shape[1]-1))
        cy=int(y*(grid.shape[0]-1))
        amp=1.0+0.3*max(int(r.get("dog_count",1)),1)
        sx=6
        sy=6
        xs=np.arange(grid.shape[1])
        ys=np.arange(grid.shape[0])
        X,Y=np.meshgrid(xs,ys)
        gauss=np.exp(-(((X-cx)**2)/(2*sx*sx)+((Y-cy)**2)/(2*sy*sy)))*amp
        grid+=gauss

    fig=plt.figure(figsize=(5.2,3.0),dpi=150)
    plt.imshow(grid,origin="lower")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    st.pyplot(fig,clear_figure=True)

# =========================================
# MAIN: load data with safe debugging
# =========================================
st.caption("Status: Loading Google Sheets CSV...")

try:
    df=load_data_from_sheet(SHEET_CSV_URL)
except Exception as e:
    st.error("Failed to load Google Sheets CSV.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

# Add severity after loading
df=add_event_rate_and_severity(df)

# Optional quick debug (comment out later)
with st.expander("Debug: Loaded data preview"):
    st.write("Columns:",list(df.columns))
    st.dataframe(df.head(5),use_container_width=True)

now=datetime.now(TZ)
today=now.date()
yesterday=(now-timedelta(days=1)).date()

# =========================================
# HEADER (title + clock)
# =========================================
st.markdown('<div class="header-bar">',unsafe_allow_html=True)
hL,hR=st.columns([4,1])
with hL:
    st.markdown('<div class="title">Stray Dog Control System</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Municipal Smart City Monitoring Dashboard</div>',unsafe_allow_html=True)
with hR:
    st.markdown(f"<div class='clock'>{now.strftime('%I:%M:%S %p')}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='date'>{now.strftime('%d/%m/%Y')}</div>",unsafe_allow_html=True)
st.markdown("</div>",unsafe_allow_html=True)

# =========================================
# SIDEBAR FILTERS
# =========================================
with st.sidebar:
    st.header("Filters")
    cameras=["All"]+sorted(df["camera_id"].dropna().unique().tolist())
    selected_camera=st.selectbox("Camera",cameras)

    locations=["All"]+sorted(df["location"].dropna().unique().tolist())
    selected_location=st.selectbox("Location",locations)

    sevs=["All","HIGH","MED","LOW"]
    selected_sev=st.selectbox("Severity",sevs)

filtered=df.copy()
if selected_camera!="All":
    filtered=filtered[filtered["camera_id"]==selected_camera]
if selected_location!="All":
    filtered=filtered[filtered["location"]==selected_location]
if selected_sev!="All":
    filtered=filtered[filtered["severity"]==selected_sev]

if filtered.empty:
    st.info("No events match the selected filters.")
    st.stop()

# =========================================
# KPI COMPUTATIONS (today vs yesterday, active alerts, active cameras)
# =========================================
df_today=df[df["timestamp"].dt.date==today]
df_yday=df[df["timestamp"].dt.date==yesterday]
today_count=len(df_today)
yday_count=len(df_yday)

delta=today_count-yday_count
if yday_count==0:
    delta_text="No baseline for yesterday"
else:
    pct=(delta/max(yday_count,1))*100.0
    arrow="▲" if delta>0 else "▼" if delta<0 else "•"
    direction="higher" if delta>0 else "lower" if delta<0 else "same"
    delta_text=f"{arrow} {abs(pct):.0f}% {direction} vs yesterday"

recent_cut=now-timedelta(minutes=ALERT_LOOKBACK_MIN)
df_recent_alerts=df[df["timestamp"]>=pd.Timestamp(recent_cut)]
active_alerts=int((df_recent_alerts["severity"].isin(["MED","HIGH"])).sum())
critical_alerts=int((df_recent_alerts["severity"]=="HIGH").sum())

all_cams=df["camera_id"].dropna().unique().tolist()
cam_total=len(all_cams)
if cam_total==0:
    oper_pct=0
else:
    last_per_cam=df.groupby("camera_id")["timestamp"].max()
    active_cams=int((last_per_cam>=pd.Timestamp(now-timedelta(seconds=ACTIVE_SEC))).sum())
    oper_pct=int(round((active_cams/max(cam_total,1))*100))

focus_loc=selected_location if selected_location!="All" else "All locations"

# =========================================
# KPI ROW (boxes under title)
# =========================================
k1,k2,k3,k4,k5=st.columns([1.2,1.2,1.2,1.2,1.2])
with k1:
    st.markdown(kpi_card("📊","Total Detections Today",f"{today_count}",delta_text),unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("🚨","Active Alerts",f"{active_alerts}",f"{critical_alerts} high severity"),unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("📍","Focus Location",f"{focus_loc}","Filtered monitoring area"),unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("📷","Active Cameras",f"{cam_total}",f"{oper_pct}% operational (last {ACTIVE_SEC}s)"),unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card("⏱️","Refresh Cycle",f"{int(REFRESH_MS/1000)}s","Auto-check for new data"),unsafe_allow_html=True)

# =========================================
# MAIN PANELS (live feed + heatmap + recent alerts)
# =========================================
left,right=st.columns([2.2,1.0])

with left:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🟢 Live AI Detection Feed <span class='badge'>LIVE</span></div>",unsafe_allow_html=True)

    latest=filtered.sort_values("timestamp").iloc[-1]
    cam_name=latest.get("camera_id","unknown")
    loc=latest.get("location","unknown")
    ts=latest.get("timestamp")
    conf=float(latest.get("confidence",0.0))
    conf_txt=f"{conf_label(conf)} ({int(round(conf*100))}%)"
    sev=latest.get("severity","LOW")
    dogs=int(latest.get("dog_count",1))

    # Optional detailed object counts if you store them
    stray_count=int(latest.get("stray_count",dogs)) if "stray_count" in filtered.columns else dogs
    pet_count=int(latest.get("pet_count",0)) if "pet_count" in filtered.columns else 0
    human_count=int(latest.get("human_count",0)) if "human_count" in filtered.columns else 0

    topA,topB=st.columns([1.2,1.0])
    with topA:
        st.markdown(f"<div class='smallbox'><b>Camera:</b> {cam_name}<br><b>Location:</b> {loc}</div>",unsafe_allow_html=True)
    with topB:
        st.markdown(f"<div class='smallbox'><b>AI:</b> Detection Active<br><b>Severity:</b> {badge_html(sev)}</div>",unsafe_allow_html=True)

    img_url=str(latest.get("image_url","")).strip()
    if img_url!="":
        st.image(img_url,use_container_width=True,caption="Latest detection snapshot")
    else:
        st.info("No image_url found. Add image_url in Google Sheets to display snapshots (Picsum/Drive/public URL).")

    bL,bR=st.columns([1.2,1.0])
    with bL:
        st.markdown(
            f"<div class='smallbox'><b>Objects:</b> {stray_count} Stray dogs, {pet_count} Pet dogs, {human_count} Humans<br><b>Confidence:</b> {conf_txt}</div>",
            unsafe_allow_html=True
        )
    with bR:
        ts_str=ts.tz_localize(None).strftime("%d/%m/%Y %I:%M:%S %p") if isinstance(ts,pd.Timestamp) else str(ts)
        st.markdown(f"<div class='smallbox'><b>Timestamp:</b><br>{ts_str}</div>",unsafe_allow_html=True)

    st.markdown("</div>",unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🔥 Detection Heatmap</div>",unsafe_allow_html=True)
    heat_cut=now-timedelta(hours=HEATMAP_LOOKBACK_HR)
    df_heat=df[df["timestamp"]>=pd.Timestamp(heat_cut)]
    if df_heat.empty:
        st.caption("No recent detections for heatmap.")
    else:
        render_heatmap(df_heat.tail(200))
        st.caption("Higher intensity indicates higher detection density (approx.).")
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:12px;">',unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🧾 Recent Alerts</div>",unsafe_allow_html=True)
    alerts=df_recent_alerts[df_recent_alerts["severity"].isin(["MED","HIGH"])].sort_values("timestamp",ascending=False).head(8)

    if alerts.empty:
        st.caption("No recent MED/HIGH alerts detected.")
    else:
        for _,r in alerts.iterrows():
            t=r["timestamp"]
            t_str=t.tz_localize(None).strftime("%H:%M:%S") if isinstance(t,pd.Timestamp) else str(t)
            sev=r.get("severity","LOW")
            cam=r.get("camera_id","")
            loc=r.get("location","")
            cnt=int(r.get("dog_count",1))
            title="Pack Detected" if cnt>=2 else "Single Detection"
            st.markdown(
                f"<div class='alert-item'>"
                f"<div class='alert-time'>{t_str}</div>"
                f"<div class='alert-title'>{title} {badge_html(sev)}</div>"
                f"<div class='alert-meta'><b>Location:</b> {loc}<br><b>Camera:</b> {cam}<br><b>Count:</b> {cnt} dog(s)</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>",unsafe_allow_html=True)

# =========================================
# EVENT LOG TABLE
# =========================================
st.markdown('<div class="panel" style="margin-top:12px;">',unsafe_allow_html=True)
st.markdown("<div class='panel-title'>📄 Event Log</div>",unsafe_allow_html=True)
show_cols=[c for c in ["timestamp","camera_id","location","class","confidence","dog_count","event_rate","severity","risk_score","image_url"] if c in filtered.columns]
st.dataframe(filtered[show_cols].tail(50).reset_index(drop=True),use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

st.markdown(f"<div class='footer'>Status: Dashboard checks Google Sheets for new data every {int(REFRESH_MS/1000)} seconds.</div>",unsafe_allow_html=True)
