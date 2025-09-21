import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import datetime
shapefile_state="source/India Shape/india_st.shp"
shapefile_district="source/India Shape/india_ds.shp"
'''
********************************************************************************************************************
'''
# date->day no
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import datetime

def plot_rainfall_gradient(csv_file,year, month, date,out_path):
    # --- 1. Load Data ---
    df = pd.read_csv(csv_file)
    rain_cols = [c for c in df.columns if c.startswith("Prec_")]

    # Parse lat/lon from column names
    lat_lon = []
    for col in rain_cols:
        parts = col.split("_")
        lat = int(parts[1]) / 100 if len(parts[1]) > 2 else float(parts[1])
        lon = int(parts[2]) / 100 if len(parts[2]) > 2 else float(parts[2])
        lat_lon.append((lat, lon))

    lats = sorted(set([p[0] for p in lat_lon]))
    lons = sorted(set([p[1] for p in lat_lon]))

    # --- 2. Select the day ---
    day_df = df[(df["Year"] == year) & (df["Month"] == month) & (df["Date"] == date)].copy()
    if day_df.empty:
        print(f"No data found for {year}-{month:02d}-{date:02d}")
        return

    rain_values = day_df[rain_cols].iloc[0].values

    # --- 3. Reshape rainfall into 2D grid (lat × lon) ---
    rain_grid = np.full((len(lats), len(lons)), np.nan)
    for (lat, lon), val in zip(lat_lon, rain_values):
        i = lats.index(lat)
        j = lons.index(lon)
        rain_grid[i, j] = val

    # --- 4. Gradient calculation ---
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    grad_lat, grad_lon = np.gradient(rain_grid, dlat, dlon)

    mag = np.sqrt(grad_lat**2 + grad_lon**2)
    u_unit = np.divide(grad_lon, mag, out=np.zeros_like(grad_lon), where=mag!=0)
    v_unit = np.divide(grad_lat, mag, out=np.zeros_like(grad_lat), where=mag!=0)

    mask = np.abs(u_unit) > 1e-3
    u_unit = np.where(mask, u_unit, np.nan)
    v_unit = np.where(mask, v_unit, np.nan)

    # --- 5. Load shapefiles ---
    state_gdf = gpd.read_file(shapefile_state)
    district_gdf = gpd.read_file(shapefile_district)

    # --- Helper: get district & state ---
    def get_district_state(point, gdf):
        match = gdf[gdf.contains(point)]
        if not match.empty:
            return match.iloc[0]["DISTRICT"], match.iloc[0]["STATE"]
        else:
            return "Unknown", "Unknown"

    # --- 6. Find rainfall maxima ---
    max_idx = np.unravel_index(np.nanargmax(rain_grid), rain_grid.shape)
    max_lat, max_lon = lats[max_idx[0]], lons[max_idx[1]]
    max_point = Point(max_lon, max_lat)
    max_district, max_state = get_district_state(max_point, district_gdf)

    # --- 7. Find next-day maxima ---
    try:
        next_day = datetime.date(year, month, date) + datetime.timedelta(days=1)
        next_df = df[(df["Year"] == next_day.year) &
                     (df["Month"] == next_day.month) &
                     (df["Date"] == next_day.day)].copy()
        if not next_df.empty:
            next_rain_values = next_df[rain_cols].iloc[0].values
            next_grid = np.full((len(lats), len(lons)), np.nan)
            for (lat, lon), val in zip(lat_lon, next_rain_values):
                i = lats.index(lat)
                j = lons.index(lon)
                next_grid[i, j] = val
            next_max_idx = np.unravel_index(np.nanargmax(next_grid), next_grid.shape)
            next_lat, next_lon = lats[next_max_idx[0]], lons[next_max_idx[1]]
            next_point = Point(next_lon, next_lat)
            next_district, next_state = get_district_state(next_point, district_gdf)
        else:
            next_lat, next_lon, next_district, next_state = None, None, "No data", "No data"
    except:
        next_lat, next_lon, next_district, next_state = None, None, "No data", "No data"

#**************************************************--- finding Inferences ---******************************************************************
    output_inference=[]
    curr_inf=f"On {year}-{month:02d}-{date:02d}, rainfall is clustered around {max_district}, {max_state} ({max_lat:.2f}, {max_lon:.2f})."
    output_inference.append(curr_inf)
    if next_lat is not None:
        upd_inf=f"On the next day ({next_day}), rainfall converges around {next_district}, {next_state} ({next_lat:.2f}, {next_lon:.2f})."
        output_inference.append(upd_inf)
#************************************************************************************************************************************************
    # --- 8. Plot ---
    fig, ax = plt.subplots(figsize=(8,8))
    cf = ax.pcolormesh(lons, lats, rain_grid, cmap="Blues", shading="auto")
    step = 1
    ax.quiver(lons[::step], lats[::step], u_unit[::step, ::step], v_unit[::step, ::step],
              color="red", scale=30, width=0.004, alpha=0.9)

    # Plot state boundaries
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

    # Mark next-day max with hollow orange circle
    if next_lat is not None:
        ax.scatter(
            next_lon, next_lat,
            s=250, facecolors="none", edgecolors="black",
            linewidths=2.5, label="Next-day"
        )

    ax.set_xlim(72.5, 81)
    ax.set_ylim(29, 37)
    ax.set_title(f"Himalayan: Rainfall + Gradient ({year}-{month:02d}-{date:02d})", fontsize=12)
    plt.colorbar(cf, ax=ax, shrink=0.7, label="Rainfall (mm)")
    ax.legend(loc="upper right")
    plt.savefig(out_path)
    plt.close()
    return output_inference