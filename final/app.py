import streamlit as st
import pandas as pd
import tempfile
import plotly.express as px 

from prediction import predict_pm25


st.set_page_config(
    page_title="PM2.5 Prediction Dashboard",
    layout="wide"
)

st.title("Hyderabad PM2.5 Prediction Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

season = st.selectbox(
    "Select Season",
    [
        "summer",
        "winter",
        "monsoon",
        "post-monsoon"
    ]
)

if st.button("Run Prediction"):

    if uploaded_file is None:
        st.error("Please upload a CSV.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        temp_csv = tmp.name

    with st.spinner("Running model..."):

        output_df = predict_pm25(
            temp_csv,
            season
        )

    st.success("Prediction completed!")

    st.subheader("Prediction Results")

    st.dataframe(output_df)

    csv = output_df.to_csv(index=False)

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name=f"predicted_pm25_{season}.csv",
        mime="text/csv"
    )

    
    st.subheader("PM2.5 Heatmap")

    fig = px.density_map(
    output_df,
    lat="latitude",
    lon="longitude",
    z="predicted_PM25",
    radius=18,
    center=dict(
        lat=17.3850,
        lon=78.4867
    ),
    zoom=11,
    map_style="carto-positron",
    color_continuous_scale=[
        "#fff5eb",
        "#fdd0a2",
        "#fdae6b",
        "#fd8d3c",
        "#e6550d",
        "#a63603"
    ],
    height=750
)

    st.plotly_chart(
        fig,
        use_container_width=True
    )