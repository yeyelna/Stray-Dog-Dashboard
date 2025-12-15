import streamlit as st
import pandas as pd
from datetime import datetime
import time,math
from streamlit_autorefresh import st_autorefresh

SHEET_CSV_URL="https://docs.google.com/spreadsheets/d/e/2PACX-1vSxyGtEAyftAfaY3M3H_sMvnA6oYcTsVjxMLVznP7SXvGA4rTXfrvzESYgSND7Z6o9qTrD-y0QRyvPo/pub?gid=0&single=true&output=csv"

REFRESH_MS=3000
WINDOW_SEC=30
MAX_EVENTS_WINDOW=10

ALERT_HIGH_SCORE=0.75
ALERT_MED_SCORE=0.45

st.set_page_config(page_title="Stray Dog Detection Dashboard (Cloud)",layout="wide")
st_autorefresh(interval=REFRESH_MS,key="data_refresh")  # refresh every 3 seconds

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
    many=trapmf(min(N/3.0,1.0),0.40,0.60,1.00,1.00)          # 2–3 dogs pushes severity up
    freq_high=trapmf(R,0.40,0.60,1.00,1.00)                  # many events in last WINDOW_SEC
    r_high=max(min(conf_high,freq_high),many)                # high risk if confident+frequent OR many dogs
    r_med=max(min(conf_med,freq_high),min(conf_high,trapmf(R,0.20,0.35,0.60,0.80)))
    r_low=max(trapmf(C,0.00,0.00,0.35,0.55),trapmf(R,0.00,0.00,0.15,0.30))
    score=(r_low*0.2+r_med*0.6+r_high*0.9)/(r_low+r_med+r_high+1e-9)
    if score>=ALERT_HIGH_SCORE:
        return "HIGH",score
    if score>=ALERT_MED_SCORE:
        return "MED",score
    return "LOW",score

def load_data_from_sheet(url):
    # Load the Google Sheet as CSV (requires the sheet to be shared/published appropriately)
    df=pd.read_csv(SHEET_CSV_URL)
    # Ensure expected columns exist; missing columns will be created with defaults
    for c in ["timestamp","camera_id","location","class","confidence","image_url","dog_count"]:
        if c not in df.columns:
            df[c]=None
    # Parse timestamps
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    df=df.dropna(subset=["timestamp"])
    df=df.sort_values("timestamp",ascending=True)
    # Coerce confidence to numeric 0..1 (YOLO confidence is usually 0..1)
    df["confidence"]=pd.to_numeric(df["confidence"],errors="coerce").fillna(0.0).clip(0,1)
    # Coerce dog_count to int; if missing, assume 1 dog for each event
    df["dog_count"]=pd.to_numeric(df["dog_count"],errors="coerce").fillna(1).astype(int)
    # Fill camera_id if missing
    df["camera_id"]=df["camera_id"].fillna("unknown").astype(str)
    # Fill location if missing
    df["location"]=df["location"].fillna("unknown").astype(str)
    # Fill class if missing
    df["class"]=df["class"].fillna("dog").astype(str)
    return df

def add_event_rate_and_severity(df,window_sec=30,max_events=10):
    # Compute a normalized recent event frequency per camera using rolling time window
    # event_rate in range [0,1] where 1 means >= max_events happened in the last window_sec
    if df.empty:
        return df
    df=df.copy().sort_values("timestamp",ascending=True)
    def _per_camera(g):
        g=g.sort_values("timestamp").copy()
        g=g.set_index("timestamp")
        cnt=g["confidence"].rolling(f"{window_sec}s").count()
        g["event_rate"]=(cnt/max_events).clip(0,1)
        return g.reset_index()
    df=df.groupby("camera_id",dropna=False,group_keys=False).apply(_per_camera)
    # Compute severity label and risk score for each row
    sev=df.apply(lambda r: pd.Series(severity_fuzzy_basic(r["confidence"],r["dog_count"],r["event_rate"])),axis=1)
    df["severity"]=sev.iloc[:,0]
    df["risk_score"]=sev.iloc[:,1]
    return df

# Load + compute severity
df=load_data_from_sheet(SHEET_CSV_URL)
df=add_event_rate_and_severity(df,window_sec=WINDOW_SEC,max_events=MAX_EVENTS_WINDOW)

# Header
last_update=datetime.now()
header_left,header_right=st.columns([3,1])
with header_left:
    st.title("Stray Dog Detection Dashboard (Cloud)")
    st.caption("Live view of stray dog detections stored in Google Sheets.")
with header_right:
    st.caption(f"Last dashboard update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("Refresh now",use_container_width=True):
        st.rerun()

# Handle empty dataset
if df.empty:
    st.info("No detection events in Google Sheets yet.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    cameras=["All"]+sorted(df["camera_id"].dropna().unique().tolist())
    selected_camera=st.selectbox("Camera",cameras)
    locations=["All"]+sorted(df["location"].dropna().unique().tolist())
    selected_location=st.selectbox("Location",locations)
    severities=["All","HIGH","MED","LOW"]
    selected_sev=st.selectbox("Severity",severities)

filtered=df.copy()
if selected_camera!="All":
    filtered=filtered[filtered["camera_id"]==selected_camera]
if selected_location!="All":
    filtered=filtered[filtered["location"]==selected_location]
if selected_sev!="All":
    filtered=filtered[filtered["severity"]==selected_sev]

if filtered.empty:
    st.warning("No events match the selected filters.")
    st.stop()

# Show latest alert banner based on severity
latest=filtered.sort_values("timestamp").iloc[-1]
alert_msg=f"Camera: {latest.get('camera_id','')} | Location: {latest.get('location','')} | Conf: {latest.get('confidence',''):.2f} | Dogs: {latest.get('dog_count','')} | Severity: {latest.get('severity','')}"
if latest.get("severity","")=="HIGH":
    st.error("⚠️ High-risk detection detected. "+alert_msg)
elif latest.get("severity","")=="MED":
    st.warning("⚠️ Detection requires attention. "+alert_msg)
else:
    st.info("Latest event: "+alert_msg)

# KPI metrics
total_detections=len(filtered)
unique_locations=filtered["location"].nunique()
last_event_time=filtered["timestamp"].max()
col1,col2,col3=st.columns(3)
with col1:
    st.metric("Total Detections",total_detections)
with col2:
    st.metric("Unique Locations",unique_locations)
with col3:
    st.metric("Last Detection Time",last_event_time.strftime("%Y-%m-%d %H:%M:%S"))

# Detections over time chart
st.subheader("Detections Over Time")
df_time=filtered.copy()
df_time["time_bucket"]=df_time["timestamp"].dt.floor("H")
detections_per_hour=df_time.groupby("time_bucket").size().reset_index(name="detections")
if not detections_per_hour.empty:
    detections_per_hour=detections_per_hour.set_index("time_bucket")
    st.line_chart(detections_per_hour)

# Optional: show latest image if you store a public URL in the sheet (image_url column)
#st.subheader("Latest Detection Evidence (optional)")
#img_url=latest.get("image_url",None)
#if isinstance(img_url,str) and img_url.strip()!="":
#    st.image(img_url,caption="Latest detection snapshot",use_container_width=True)
#else:
#    st.caption("No image_url provided in the event log.")
st.subheader("Latest Detection Evidence (optional)")
latest=filtered.sort_values("timestamp").iloc[-1]  # get latest after filters applied
img_url=latest.get("image_url","")
if isinstance(img_url,str) and img_url.strip()!="":
    st.image(img_url,caption="Latest detection snapshot",use_container_width=True)
else:
    st.caption("No image_url provided in the event log.")



# Event table
st.subheader("Recent Detection Events")
show_cols=[c for c in ["timestamp","camera_id","location","class","confidence","dog_count","event_rate","severity","risk_score"] if c in filtered.columns]
st.dataframe(filtered[show_cols].tail(50).reset_index(drop=True),use_container_width=True)

st.caption(f"Status: Dashboard checks Google Sheets for new data every {int(REFRESH_MS/1000)} seconds.")
