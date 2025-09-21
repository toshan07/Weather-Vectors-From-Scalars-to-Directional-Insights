import streamlit as st
st.set_page_config(
    page_title="Weather Vectors App",
    page_icon="🌍",
    layout="wide"
)
with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🌍 Weather Vectors: From Scalars to Directional Insights")
st.write(
    "Turning rainfall, wind, and pollution data into *directional insights* "
    "that are easier to understand, compare, and act upon."
)

st.markdown("---")

st.markdown(
    """
    ## Why This Matters
    Weather datasets often report *scalar values* (mm of rainfall, µg/m³ of pollution, etc.),  
    but they miss the *direction of change* – whether it's spreading north, intensifying westward, or drifting into cities.  

    *Direction matters.*  
    It’s the difference between raw data and actionable early warnings for *climate resilience and disaster management*.
    """
)

st.markdown(
    """
    ## Our Approach
    - Upload data for  Rainfall,  Wind, or  Pollution.  
    - We compute *gradient fields* – vectors that show where and how fast conditions are changing.  
    - Visualize them as *maps, arrows, and animations*.  

    This adds a missing dimension to weather communication:  
    From static numbers → to *dynamic, intuitive stories* about the atmosphere.
    """
)

st.markdown("---")

# --- Quick Navigation ---
st.subheader("🔎 Explore Modules")
cols = st.columns(3)

with cols[0]:
    st.markdown("### Rainfall")
    st.write("Upload rainfall datasets and see rainfall spread direction and intensity.")

with cols[1]:
    st.markdown("### Wind")
    st.write("Analyze wind fields and visualize directional flow vectors.")

with cols[2]:
    st.markdown("### Pollution")
    st.write("Track pollutants (PM2.5, PM10, SO₂, Prec) and their drift over time.")

st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: relative;
        bottom: 0;
        width: 100%;
        padding: 10px;
        text-align: center;
        color: #666;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        🌍 Developed by <b>Group 1</b> • SDA Hackathon • Weather Vectors
    </div>
    """,
    unsafe_allow_html=True
)