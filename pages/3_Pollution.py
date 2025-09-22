import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import os
import numpy as np
from source.pollution import plot_pollution_gradient,animate_pollution_gradient_year
import tempfile

st.set_page_config(page_title="Pollution Analysis", page_icon="🌫️", layout="wide")
st.header("🌫️ Pollution Data Upload & Visualization")

particles = ["PM2.5", "PM10", "SO2", "PREC"]

# -------------------
# Helpers
# -------------------
@st.cache_data
def save_uploaded_file(uploaded_file, prefix):
    suffix = os.path.splitext(uploaded_file.name)[1] 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix+"_") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name 

@st.cache_data
def cached_plot_month(datafile, year, month, particle, out_path):
    """Generate pollution gradient for given month/particle (cached)."""
    inferences = plot_pollution_gradient(
        datafile=datafile,
        year=year,
        month=month,
        particle=particle,
        step=8,
        out_path=out_path,
        cmap="RdYlGn_r"
    )
    return out_path, inferences

@st.cache_data
def cached_animate_pollution_gradient_year(data_path,year,pollutant,out_gif):
    inferences = animate_pollution_gradient_year(
        datafile=data_path,
        year=year,
        particle=pollutant,
        out_gif=out_gif,
        step=8,
        smooth=True,
        cmap="RdYlGn_r",
        smooth_frames=15
    )
    return inferences

# -------------------
# FRONTEND
# -------------------
mode = st.radio(
    "Choose visualization mode:",
    [
        "🌦️ Single Month (one pollutant)",
        "📅 Multi-Months (one pollutant)",
        "🧪 Compare Pollutants",
        "🎞️ Yearly GIF Visualization"
    ]
)

# --- Single Month ---
if mode == "🌦️ Single Month (one pollutant)":
    selected_date = st.date_input("Select Year and Month", value=date(2022, 1, 1))
    pollutant = st.selectbox("Select Pollutant", particles)
    file = st.file_uploader(f"Upload dataset for {pollutant}", type=["csv", "nc", "tif"])
    generate_btn = st.button("🚀 Generate Visualization")

    if generate_btn and file:
        os.makedirs("plots", exist_ok=True)
        data_path = save_uploaded_file(file, pollutant)
        out_path = f"plots/{pollutant}_{selected_date.strftime('%Y_%m')}.png"

        with st.spinner("⏳ Generating pollution visualization..."):
            out_path, inferences = cached_plot_month(
                datafile=data_path,
                year=selected_date.year,
                month=selected_date.month,
                particle=pollutant,
                out_path=out_path
            )

        st.success("✅ Visualization ready!")
        st.image(out_path, caption=f"{pollutant} in {selected_date.strftime('%B %Y')}", use_container_width=True)
        with st.expander("📊 Inferences"):
            for inf in inferences:
                st.markdown(f"- {inf}")

# --- Multi Months ---
elif mode == "📅 Multi-Months (one pollutant)":
    start_date = st.date_input("Start Year and Month", value=date(2022, 1, 1))
    num_months = st.number_input("Number of Months", min_value=1, max_value=12, value=3, step=1)
    pollutant = st.selectbox("Select Pollutant", particles)
    file = st.file_uploader(f"Upload dataset for {pollutant}", type=["csv", "nc", "tif"])
    generate_btn = st.button("🚀 Generate Multi-Month Visualization")

    if generate_btn and file:
        os.makedirs("plots", exist_ok=True)
        data_path = save_uploaded_file(file, pollutant)

        images_and_inferences = []
        current = start_date

        with st.spinner(f"⏳ Generating {pollutant} visualizations for {num_months} months..."):
            for i in range(num_months):
                out_path = f"plots/{pollutant}_{current.strftime('%Y_%m')}.png"
                out_path, inferences = cached_plot_month(
                    datafile=data_path,
                    year=current.year,
                    month=current.month,
                    particle=pollutant,
                    out_path=out_path
                )
                images_and_inferences.append((current, out_path, inferences))
                current += relativedelta(months=1)

        st.success("✅ All visualizations generated!")

        for i in range(0, len(images_and_inferences), 2):
            cols = st.columns(2)
            for j, (month_date, img_path, inferences) in enumerate(images_and_inferences[i:i+2]):
                with cols[j]:
                    st.image(img_path, caption=f"{pollutant} in {month_date.strftime('%B %Y')}", use_container_width=True)
                    with st.expander("📊 Inferences"):
                        for inf in inferences:
                            st.markdown(f"- {inf}")

# --- Compare Pollutants ---
elif mode == "🧪 Compare Pollutants":
    compare_mode = st.radio("Comparison type:", ["Single Month", "Multi-Months"])

    st.subheader("📂 Upload dataset for each pollutant")
    pollutant_files = {}
    for p in particles:
        pollutant_files[p] = st.file_uploader(f"Upload {p} dataset", type=["csv", "nc", "tif"], key=p)

    if compare_mode == "Single Month":
        selected_date = st.date_input("Select Year and Month", value=date(2022, 1, 1))
        generate_btn = st.button("🚀 Compare Pollutants for Single Month")

        if generate_btn and all(pollutant_files.values()):
            os.makedirs("plots", exist_ok=True)
            plots = []
            with st.spinner(f"⏳ Comparing pollutants for {selected_date.strftime('%B %Y')}..."):
                for p in particles:
                    data_path = save_uploaded_file(pollutant_files[p], p)
                    out_path = f"plots/{p}_{selected_date.strftime('%Y_%m')}.png"
                    out_path, inferences = cached_plot_month(
                        datafile=data_path,
                        year=selected_date.year,
                        month=selected_date.month,
                        particle=p,
                        out_path=out_path
                    )
                    plots.append((p, out_path, inferences))

            st.success("✅ Comparison ready!")
            for i in range(0, len(plots), 2):
                cols = st.columns(2)
                for j, (p, img_path, inferences) in enumerate(plots[i:i+2]):
                    with cols[j]:
                        st.image(img_path, caption=f"{p} in {selected_date.strftime('%B %Y')}", use_container_width=True)
                        with st.expander("📊 Inferences"):
                            for inf in inferences:
                                st.markdown(f"- {inf}")

    else:  # Multi-Months compare
        start_date = st.date_input("Start Year and Month", value=date(2022, 1, 1))
        num_months = st.number_input("Number of Months", min_value=1, max_value=12, value=3, step=1)
        generate_btn = st.button("🚀 Compare Pollutants Across Multiple Months")

        if generate_btn and all(pollutant_files.values()):
            os.makedirs("plots", exist_ok=True)
            current = start_date
            with st.spinner(f"⏳ Comparing pollutants across {num_months} months..."):
                for i in range(num_months):
                    month_plots = []
                    for p in particles:
                        data_path = save_uploaded_file(pollutant_files[p], p)
                        out_path = f"plots/{p}_{current.strftime('%Y_%m')}.png"
                        out_path, inferences = cached_plot_month(
                            datafile=data_path,
                            year=current.year,
                            month=current.month,
                            particle=p,
                            out_path=out_path
                        )
                        month_plots.append((p, out_path, inferences))

                    with st.expander(f"📅 {current.strftime('%B %Y')}"):
                        for j in range(0, len(month_plots), 2):
                            cols = st.columns(2)
                            for k, (p, img_path, inferences) in enumerate(month_plots[j:j+2]):
                                with cols[k]:
                                    st.image(img_path, caption=f"{p} in {current.strftime('%B %Y')}", use_container_width=True)
                                    with st.expander("📊 Inferences"):
                                        for inf in inferences:
                                            st.markdown(f"- {inf}")
                    current += relativedelta(months=1)
            st.success("✅ All comparisons generated!")

# --- Yearly GIF Visualization ---
elif mode == "🎞️ Yearly GIF Visualization":
    list_of_years=list(np.arange(2010,2024))
    year = st.selectbox("Select Year", list_of_years)  # adjust as needed
    pollutant = st.selectbox("Select Pollutant", particles)
    file = st.file_uploader(f"Upload yearly dataset for {pollutant}", type=["csv", "nc", "tif"])
    generate_btn = st.button("🚀 Generate Yearly GIF")

    if generate_btn and file:
        os.makedirs("plots", exist_ok=True)
        data_path = save_uploaded_file(file, pollutant)
        out_gif = f"plots/{pollutant}_{year}.gif"

        with st.spinner(f"⏳ Generating {pollutant} GIF for {year}..."):
            inferences = cached_animate_pollution_gradient_year(
                data_path=data_path,
                year=year,
                pollutant=pollutant,
                out_gif=out_gif
            )

        st.success("✅ GIF ready!")
        st.image(out_gif, caption=f"{pollutant} trends in {year}", use_container_width=True)
        with st.expander("📊 Inferences"):
            for inf in inferences:
                st.markdown(f"- {inf}")