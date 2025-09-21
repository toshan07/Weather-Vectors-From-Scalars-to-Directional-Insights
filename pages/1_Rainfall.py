import streamlit as st
from datetime import date, timedelta
import os
from source.rain import plot_rainfall_gradient

st.set_page_config(page_title="Rainfall Analysis", page_icon="🌧️", layout="wide")
st.header("🌧️ Rainfall Data Upload & Visualization")

# -------------------
# CACHED FUNCTIONS
# -------------------

@st.cache_data
def save_uploaded_file(uploaded_file):
    """Save uploaded file locally and return the path."""
    data_path = "rainfall_input.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.read())
    return data_path

@st.cache_data
def cached_plot_single_day(datafile, year, month, day, out_path):
    """Generate rainfall gradient for a single day (cached)."""
    if not os.path.exists(out_path):
        inferences = plot_rainfall_gradient(
            csv_file=datafile,
            year=year,
            month=month,
            date=day,
            out_path=out_path
        )
    else:
        # If file exists, still get inferences again
        inferences = plot_rainfall_gradient(
            csv_file=datafile,
            year=year,
            month=month,
            date=day,
            out_path=out_path
        )
    return out_path, inferences

# -------------------
# FRONTEND
# -------------------

# File upload
rainfall_file = st.file_uploader("Upload rainfall dataset", type=["csv", "nc", "tif"])

# Option selector
mode = st.radio("Choose visualization mode:", ["🌦️ Single Day", "📅 Multi-Day"])

# Single-day input
if mode == "🌦️ Single Day":
    selected_date = st.date_input("Select Date", value=date(2022, 1, 1))
    generate_btn = st.button("🚀 Submit & Generate Visualization")

    if generate_btn and rainfall_file:
        os.makedirs("plots", exist_ok=True)

        # Save file locally (cached)
        data_path = save_uploaded_file(rainfall_file)

        out_path = "plots/rainfall.png"
        with st.spinner("⏳ Generating rainfall visualization..."):
            out_path, inferences = cached_plot_single_day(
                datafile=data_path,
                year=selected_date.year,
                month=selected_date.month,
                day=selected_date.day,
                out_path=out_path
            )

        st.success("✅ Visualization ready!")
        st.image(out_path, caption=f"Rainfall gradient on {selected_date}", use_container_width=True)

        st.subheader("📊 Inferences")
        for inf in inferences:
            st.markdown(f"- {inf}")

# Multi-day input
else:
    start_date = st.date_input("Start Date", value=date(2022, 1, 1))
    num_days = st.number_input("Number of Days", min_value=1, max_value=30, value=3, step=1)
    generate_btn = st.button("🚀 Submit & Generate Multi-Day Visualization")

    if generate_btn and rainfall_file:
        os.makedirs("plots", exist_ok=True)

        # Save file locally (cached)
        data_path = save_uploaded_file(rainfall_file)

        images_and_inferences = []

        with st.spinner(f"⏳ Generating rainfall visualization for {num_days} days..."):
            for i in range(num_days):
                current_date = start_date + timedelta(days=i)
                out_path = f"plots/rainfall_{current_date.strftime('%Y%m%d')}.png"

                out_path, inferences = cached_plot_single_day(
                    datafile=data_path,
                    year=current_date.year,
                    month=current_date.month,
                    day=current_date.day,
                    out_path=out_path
                )
                images_and_inferences.append((current_date, out_path, inferences))

        st.success("✅ All visualizations generated!")

        # Display images in grid (2 per row) with inferences
        for i in range(0, len(images_and_inferences), 2):
            cols = st.columns(2)
            for j, (current_date, img_path, inferences) in enumerate(images_and_inferences[i:i+2]):
                with cols[j]:
                    st.image(img_path, caption=f"Rainfall on {current_date}", use_container_width=True)
                    st.markdown("**📊 Inferences:**")
                    for inf in inferences:
                        st.markdown(f"- {inf}")
