import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import calendar
import os

from source.wind import animate_wind_vectors, plot_wind_vectors_single_day
from source.inferences import generate_infernces_wind

# -------------------
# CACHE HELPERS
# -------------------

@st.cache_data
def save_uploaded_file(uploaded_file):
    """Save uploaded file locally and return the path."""
    data_path = "wind_input.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.read())
    return data_path

@st.cache_data
def cached_plot_single_day(datafile, year, month, day, out_path):
    """Generate wind vectors for a single day + inferences (cached)."""
    plot_wind_vectors_single_day(
        datafile=datafile,
        year=year,
        month=month,
        day=day,
        out_path=out_path,
        step=1,
        scale=85,
        equal_length=False,
    )
    return out_path

@st.cache_data
def cached_animate_month(datafile, year, month, num_days, step, scale, equal_length, smooth_frames, out_file):
    """Generate monthly wind GIF + inferences (cached)."""
    animate_wind_vectors(
        datafile=datafile,
        year=year,
        month=month,
        tot_days=num_days,
        step=step,
        scale=scale,
        equal_length=equal_length,
        smooth_frames=smooth_frames,
        out_file=out_file,
    )
    inferences = generate_infernces_wind(
        out_file
    )
    return out_file, inferences


# -------------------
# FRONTEND
# -------------------

st.set_page_config(page_title="Wind Analysis", page_icon="💨", layout="wide")
st.header("💨 Wind Data Upload & Visualization")

# File upload
wind_file = st.file_uploader("Upload wind dataset", type=["csv", "nc", "tif"])

# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2022, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date(2022, 1, 2))

# Submit button
generate_btn = st.button("🚀 Submit & Generate Visualization")

# Run only if file + button
if generate_btn and wind_file:
    os.makedirs("plots", exist_ok=True)
    st.success("✅ File uploaded successfully!")

    # Save uploaded file
    data_path = save_uploaded_file(wind_file)

    # Single-day case
    if start_date == end_date:
        out_path = "plots/wind_single.png"
        with st.spinner("⏳ Generating visualization..."):
            out_path= cached_plot_single_day(
                datafile=data_path,
                year=start_date.year,
                month=start_date.month,
                day=start_date.day,
                out_path=out_path
            )
        st.success("✅ Visualization ready!")
        st.image(out_path, caption=f"Wind vectors on {start_date.strftime('%d %B %Y')}", use_container_width=True)
        
    # Multi-day/month case
    else:
        st.info("📅 Generating monthly GIFs for the selected time range...")

        current = start_date.replace(day=1)
        images_and_inferences = []

        with st.spinner("⏳ Generating monthly visualizations..."):
            while current <= end_date:
                year, month = current.year, current.month
                days_in_month = calendar.monthrange(year, month)[1]

                month_start = max(current, start_date)
                month_end = min(current.replace(day=days_in_month), end_date)
                num_days = (month_end - month_start).days + 1

                out_gif = f"plots/wind_{year}_{month:02d}.gif"
                out_gif, inferences = cached_animate_month(
                    datafile=data_path,
                    year=year,
                    month=month,
                    num_days=num_days,
                    step=1,
                    scale=85,
                    equal_length=False,
                    smooth_frames=5,
                    out_file=out_gif,
                )
                images_and_inferences.append((month_start, month_end, out_gif, inferences))
                current += relativedelta(months=1)

        st.success("✅ All monthly visualizations generated!")

        # Display in grid (2 per row)
        for i in range(0, len(images_and_inferences), 2):
            cols = st.columns(2)
            for j, (month_start, month_end, gif_path, inferences) in enumerate(images_and_inferences[i:i+2]):
                with cols[j]:
                    st.subheader(f"🌬️ {month_start.strftime('%B %Y')}")
                    st.image(
                        gif_path,
                        caption=f"Wind animation ({month_start.strftime('%d %B')} to {month_end.strftime('%d %B')})",
                        use_container_width=True
                    )
                    with st.expander("📊 Inferences"):
                        for inf in inferences:
                            st.markdown(f"- {inf}")