import streamlit as st
import pandas as pd
import os
from datetime import datetime
st.set_page_config(page_title="Stray Dog Detection Dashboard",layout="wide")
st.set_page_config(page_title="Stray Dog Detection Dashboard",layout="wide")
st.autorefresh(interval=3000,key="refresh")  # refresh every 3 seconds
EVENT_FILE="events.csv"
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp","camera_id","location","class","confidence","image_path"])
    df=pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    df=df.dropna(subset=["timestamp"])
    df=df.sort_values("timestamp",ascending=True)
    return df
if hasattr(st,"autorefresh"):
    st.autorefresh(interval=3000,key="data_refresh")
df=load_data(EVENT_FILE)
st.title("Stray Dog Detection Dashboard")
st.caption("Real-time monitoring of stray dog detections using YOLO and CCTV/video streams.")
if df.empty:
    st.info("No detection events found yet. Make sure your detection script is writing to 'events.csv'.")
    st.stop()
with st.sidebar:
    st.header("Filters")
    cameras=["All"]
    if "camera_id" in df.columns:
        cameras+=sorted([c for c in df["camera_id"].dropna().unique()])
    selected_camera=st.selectbox("Camera",cameras)
    locations=["All"]
    if "location" in df.columns:
        locations+=sorted([l for l in df["location"].dropna().unique()])
    selected_location=st.selectbox("Location",locations)
filtered=df.copy()
if selected_camera!="All":
    filtered=filtered[filtered["camera_id"]==selected_camera]
if selected_location!="All":
    filtered=filtered[filtered["location"]==selected_location]
if filtered.empty:
    st.warning("No events match the selected filters.")
    st.stop()
total_detections=len(filtered)
unique_locations=filtered["location"].nunique() if "location" in filtered.columns else 0
last_event_time=filtered["timestamp"].max()
last_event_time_str=last_event_time.strftime("%Y-%m-%d %H:%M:%S")
col1,col2,col3=st.columns(3)
with col1:
    st.metric("Total Detections",total_detections)
with col2:
    st.metric("Unique Locations",unique_locations)
with col3:
    st.metric("Last Detection Time",last_event_time_str)
time_chart_container,latest_image_container=st.columns([2,1])
with time_chart_container:
    st.subheader("Detections Over Time")
    df_time=filtered.copy()
    df_time["time_bucket"]=df_time["timestamp"].dt.floor("H")
    detections_per_hour=df_time.groupby("time_bucket").size().reset_index(name="detections")
    detections_per_hour=detections_per_hour.set_index("time_bucket")
    st.line_chart(detections_per_hour)
with latest_image_container:
    st.subheader("Latest Detection")
    latest_row=filtered.iloc[-1]
    img_path=latest_row.get("image_path",None)
    if isinstance(img_path,str) and os.path.exists(img_path):
        st.image(img_path,use_column_width=True,caption=f"Camera: {latest_row.get('camera_id','')} | Location: {latest_row.get('location','')} | Confidence: {latest_row.get('confidence','')}")
    else:
        st.info("Latest detection image not available or file not found.")
st.subheader("Recent Detection Events")
show_cols=[c for c in ["timestamp","camera_id","location","class","confidence"] if c in filtered.columns]
st.dataframe(filtered[show_cols].tail(50).reset_index(drop=True))
