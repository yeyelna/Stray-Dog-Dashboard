import streamlit as st
import pandas as pd
import os
from datetime import datetime,timedelta
import random
st.set_page_config(page_title="Stray Dog Detection Dashboard (Testing)",layout="wide")
def generate_dummy_data(n=100):
    now=datetime.now()
    cameras=["cam1","cam2","cam3"]
    locations=["Street A","Street B","Playground 1","Market Zone"]
    rows=[]
    for i in range(n):
        ts=now-timedelta(minutes=random.randint(0,300))
        cam=random.choice(cameras)
        loc=random.choice(locations)
        cls="dog"
        conf=round(random.uniform(0.5,0.99),2)
        rows.append({"timestamp":ts,"camera_id":cam,"location":loc,"class":cls,"confidence":conf,"image_path":""})
    df=pd.DataFrame(rows)
    df=df.sort_values("timestamp",ascending=True).reset_index(drop=True)
    return df
df=generate_dummy_data(120)
st.title("Stray Dog Detection Dashboard (Testing Mode)")
st.caption("This dashboard shows randomly generated detection events to preview the layout and components.")
with st.sidebar:
    st.header("Filters")
    cameras=["All"]
    cameras+=sorted([c for c in df["camera_id"].dropna().unique()])
    selected_camera=st.selectbox("Camera",cameras)
    locations=["All"]
    locations+=sorted([l for l in df["location"].dropna().unique()])
    selected_location=st.selectbox("Location",locations)
filtered=df.copy()
if selected_camera!="All":
    filtered=filtered[filtered["camera_id"]==selected_camera]
if selected_location!="All":
    filtered=filtered[filtered["location"]==selected_location]
if filtered.empty:
    st.warning("No dummy events match the selected filters. Try changing the filters.")
    st.stop()
total_detections=len(filtered)
unique_locations=filtered["location"].nunique()
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
    st.subheader("Detections Over Time (Dummy Data)")
    df_time=filtered.copy()
    df_time["time_bucket"]=df_time["timestamp"].dt.floor("H")
    detections_per_hour=df_time.groupby("time_bucket").size().reset_index(name="detections")
    detections_per_hour=detections_per_hour.set_index("time_bucket")
    st.line_chart(detections_per_hour)
with latest_image_container:
    st.subheader("Latest Detection (Dummy)")
    latest_row=filtered.iloc[-1]
    st.write(f"Camera: {latest_row.get('camera_id','')}")
    st.write(f"Location: {latest_row.get('location','')}")
    st.write(f"Confidence: {latest_row.get('confidence','')}")
    st.info("No image displayed in testing mode. In the real dashboard this will show the latest snapshot.")
st.subheader("Recent Detection Events (Dummy Data)")
show_cols=[c for c in ["timestamp","camera_id","location","class","confidence"] if c in filtered.columns]
st.dataframe(filtered[show_cols].tail(50).reset_index(drop=True))
