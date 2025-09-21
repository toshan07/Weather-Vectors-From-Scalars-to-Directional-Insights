import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d

# --- Load shapefile ---
state_gdf = gpd.read_file("source/India Shape/india_st.shp")

'''
********************************************************************************************************************
'''
def prepare_wind_data(datafile, year, month, day, equal_length=False):
    """Return U, V, lon_mesh, lat_mesh for a given day."""
    df = pd.read_csv(datafile)

    ws_cols = [c for c in df.columns if c.startswith("WS10M_")]
    wd_cols = [c for c in df.columns if c.startswith("WD10M_")]

    coords = [c.split("_", 1)[1] for c in ws_cols]
    lat_lon = []
    for c in coords:
        parts = c.split("_")
        lat = int(parts[0]) / 100 if len(parts[0]) > 2 else float(parts[0])
        lon = int(parts[1]) / 100 if len(parts[1]) > 2 else float(parts[1])
        lat_lon.append((lat, lon))

    lats = sorted(set([p[0] for p in lat_lon]))
    lons = sorted(set([p[1] for p in lat_lon]))

    mask = (df["Year"] == year) & (df["Month"] == month) & (df["Date"] == day)
    if not mask.any():
        return None, None, None, None

    row = df.loc[mask].iloc[0]
    ws_values = row[ws_cols].values
    wd_values = row[wd_cols].values

    theta = np.deg2rad(wd_values)
    if equal_length:
        u = -np.sin(theta)
        v = -np.cos(theta)
    else:
        u = -ws_values * np.sin(theta)
        v = -ws_values * np.cos(theta)

    U = np.full((len(lats), len(lons)), np.nan)
    V = np.full((len(lats), len(lons)), np.nan)
    for (lat, lon), uu, vv in zip(lat_lon, u, v):
        i = lats.index(lat)
        j = lons.index(lon)
        U[i, j] = uu
        V[i, j] = vv

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    return U, V, lon_mesh, lat_mesh

def animate_wind_vectors(datafile, year, month,tot_days, step=3, scale=300, equal_length=False, smooth_frames=10, out_file="wind_fade.gif"):
    """
    Create a GIF of wind vectors for a month with smooth transitions and fade effect.
    """
    # --- Collect all daily data ---
    daily_data = []
    for day in range(1, tot_days+1):  # try all days
        try:
            U, V, lon_mesh, lat_mesh = prepare_wind_data(datafile, year, month, day, equal_length)
            if U is not None:
                daily_data.append((day, U, V))
        except Exception:
            continue

    if not daily_data:
        raise ValueError("No data found for this month!")

    # --- Interpolation (for smoother fade) ---
    days = [d for d, _, _ in daily_data]
    U_series = np.array([U for _, U, _ in daily_data])
    V_series = np.array([V for _, _, V in daily_data])

    # Number of animation frames
    total_frames = (len(days) - 1) * smooth_frames  

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10,10))
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

    lat_min, lat_max = 29, 37
    lon_min, lon_max = 72.5, 81
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = ax.set_title("")

    # Initialize with first day's vectors
    U0, V0 = U_series[0], V_series[0]
    speed0 = np.sqrt(U0**2 + V0**2)
    Q = ax.quiver(
        lon_mesh[::step, ::step], lat_mesh[::step, ::step],
        U0[::step, ::step], V0[::step, ::step],
        speed0[::step, ::step],
        cmap="viridis", scale=scale, width=0.004, alpha=1.0
    )
    cbar = fig.colorbar(Q, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Wind Speed (m/s)")

    def update(frame):
        # Which two days we are interpolating between
        day_idx = frame // smooth_frames
        t = (frame % smooth_frames) / smooth_frames  # interpolation factor

        if day_idx >= len(days) - 1:
            return Q, title

        U1, V1 = U_series[day_idx], V_series[day_idx]
        U2, V2 = U_series[day_idx+1], V_series[day_idx+1]

        # Linear interpolation
        Uf = (1 - t) * U1 + t * U2
        Vf = (1 - t) * V1 + t * V2
        speed = np.sqrt(Uf**2 + Vf**2)

        Q.set_UVC(Uf[::step, ::step], Vf[::step, ::step], speed[::step, ::step])
        title.set_text(f"Wind Vectors - {year}-{month:02d}, Day {days[day_idx]} → {days[day_idx+1]}")

        return Q, title

    anim = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    anim.save(out_file, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"✅ GIF saved {out_file}")

'''
********************************************************************************************************************
'''

def plot_wind_vectors_single_day(datafile, year, month, day, out_path,step=3, scale=300, equal_length=False):
    """
    Plot wind vectors at 10m for a given date with color-coded magnitude + streamlines.
    """
    # --- 1. Load Data ---
    df = pd.read_csv(datafile)
    ws_cols = [c for c in df.columns if c.startswith("WS10M_")]
    wd_cols = [c for c in df.columns if c.startswith("WD10M_")]

    coords = [c.split("_", 1)[1] for c in ws_cols]
    lat_lon = []
    for c in coords:
        parts = c.split("_")
        lat = int(parts[0]) / 100 if len(parts[0]) > 2 else float(parts[0])
        lon = int(parts[1]) / 100 if len(parts[1]) > 2 else float(parts[1])
        lat_lon.append((lat, lon))

    lats = sorted(set([p[0] for p in lat_lon]))
    lons = sorted(set([p[1] for p in lat_lon]))

    # --- 2. Select the requested day ---
    mask = (df["Year"] == year) & (df["Month"] == month) & (df["Date"] == day)
    if not mask.any():
        raise ValueError(f"No data found for {year}-{month:02d}-{day:02d}")

    row = df.loc[mask].iloc[0]
    ws_values = row[ws_cols].values
    wd_values = row[wd_cols].values

    # --- 3. Convert to vector components ---
    theta = np.deg2rad(wd_values)
    if equal_length:
        u = -np.sin(theta)
        v = -np.cos(theta)
        magnitude = ws_values
    else:
        u = -ws_values * np.sin(theta)
        v = -ws_values * np.cos(theta)
        magnitude = np.sqrt(u**2 + v**2)

    # --- 4. Reshape into grid ---
    U = np.full((len(lats), len(lons)), np.nan)
    V = np.full((len(lats), len(lons)), np.nan)
    M = np.full((len(lats), len(lons)), np.nan)
    for (lat, lon), uu, vv, mm in zip(lat_lon, u, v, magnitude):
        i = lats.index(lat)
        j = lons.index(lon)
        U[i, j] = uu
        V[i, j] = vv
        M[i, j] = mm

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # --- 5. Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

    # Quiver (arrows)
    Q = ax.quiver(
        lon_mesh[::step, ::step], lat_mesh[::step, ::step],
        U[::step, ::step], V[::step, ::step],
        M[::step, ::step],  # color by magnitude
        cmap="plasma", scale=scale, width=0.004, alpha=0.9
    )

    # Streamlines (smooth flow lines)
    strm = ax.streamplot(
        lon_mesh, lat_mesh, U, V,
        color=M, cmap="plasma", linewidth=1, density=1.2, arrowsize=1
    )

    # Colorbar (shared for quiver + streamlines)
    cbar = fig.colorbar(Q, ax=ax, shrink=0.7)
    cbar.set_label("Wind Speed (m/s)")

    # Reference arrow
    ax.quiverkey(Q, X=0.9, Y=1.05, U=10, label="10 m/s", labelpos="E")

    # Bounding box (Northern India region)
    lat_min, lat_max = 29, 37
    lon_min, lon_max = 72.5, 81
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title(f"Wind Vectors + Streamlines at 10m ({year}-{month:02d}-{day:02d})", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.savefig(out_path)
    plt.close()
'''
********************************************************************************************************************
'''