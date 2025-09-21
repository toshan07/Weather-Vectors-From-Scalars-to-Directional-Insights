import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from shapely.geometry import Point
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl

from source.inferences import generate_infernces_pollution

district_file="source/India Shape/india_ds.shp"
state_gdf = gpd.read_file("source/India Shape/india_st.shp")
india_union = state_gdf.unary_union  # merged India polygon

def generate_inference(u_unit_masked, v_unit_masked, mag, mask, region_name, 
                       grid_lon, grid_lat, grid_pollutant, district_shapefile):
    """
    Parameters
    ----------
    u_unit_masked : 2D array
        X-component of unit vectors (masked already).
    v_unit_masked : 2D array
        Y-component of unit vectors (masked already).
    mag : 2D array
        Gradient magnitude (not masked).
    mask : 2D boolean array
        Valid data mask.
    region_name : str
        Name of region (for reporting).
    grid_lon, grid_lat : 2D arrays
        Longitude and latitude meshgrid.
    grid_pollutant : 2D array
        Pollutant values (masked already).
    district_shapefile : str
        Path to district shapefile (with DISTRICT column).
    
    Returns
    -------
    str
        Human-readable summary.
    """
    # ========================
    # 1. Gradient direction
    # ========================
    mag_masked = np.where(mask, mag, np.nan)
    u_mean = np.nanmean(u_unit_masked)
    v_mean = np.nanmean(v_unit_masked)

    if np.isnan(u_mean) or np.isnan(v_mean):
        return f"No clear pollution movement detected in {region_name}."

    # Compute dominant gradient direction
    angle = np.degrees(np.arctan2(v_mean, u_mean))
    angle = (angle + 360) % 360
    compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    direction = compass[idx]

    avg_mag = np.nanmean(mag_masked)
    max_mag = np.nanmax(mag_masked)

    s1=f"In {region_name}, pollution flow is predominantly {direction}."
    s2=f"Average gradient strength is {avg_mag:.2f} with a maximum of {max_mag:.2f}."

    if avg_mag < 0.2:
        s3= "Overall movement is weak, suggesting stagnation of pollutants."
    elif avg_mag < 0.5:
        s3= "Moderate movement indicates gradual spreading."
    else:
        s3= "Strong gradient suggests rapid pollutant transport."
    summary=[s1,s2,s3]

    # ========================
    # 2. District-level summary
    # ========================
    gdf_districts = gpd.read_file(district_shapefile)
    district_col = "DISTRICT"  # use the correct column from your shapefile

    points = [Point(x, y) for x, y in zip(grid_lon.flatten(), grid_lat.flatten())]
    values = grid_pollutant.flatten()
    valid_mask = ~np.isnan(values)

    gdf_points = gpd.GeoDataFrame(
        {"pollutant": values[valid_mask]},
        geometry=np.array(points)[valid_mask],
        crs=gdf_districts.crs
    )

    joined = gpd.sjoin(gdf_points, gdf_districts, how="left", predicate="within")

    district_means = joined.groupby(district_col)["pollutant"].mean().dropna()
    if len(district_means) > 0:
        top_districts = district_means.sort_values(ascending=False).head(2)
        low_districts = district_means.sort_values().head(2)
        s4=f"Highest pollutant levels are observed in {', '.join(top_districts.index)}."
        s5=f"Lowest levels are seen in {', '.join(low_districts.index)}."
        summary.append(s4)
        summary.append(s5)

    return summary


def plot_pollution_gradient(datafile, year, month,particle, out_path, step=3, smooth=True,
                            cmap="RdYlGn_r"):

    # --- 1. Load Data ---
    df = pd.read_csv(datafile)

    colname = f"{year}-{month:02d}"
    if colname not in df.columns:
        raise ValueError(f"No column {colname} in dataset")

    points = df[["lat", "lon"]].values
    values = np.log10(df[colname].values + 1e-12) 

    # --- 2. Grid setup (bounding box) ---
    lat_min, lat_max = 29, 37
    lon_min, lon_max = 72.5, 81

    grid_lat = np.linspace(lat_min, lat_max, 200)
    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # --- 3. Interpolate pollution ---
    grid_poll = griddata(points, values, (grid_lat_mesh, grid_lon_mesh), method="linear")
    if smooth:
        grid_poll = gaussian_filter(grid_poll, sigma=2, mode="nearest")

    # --- 4. Mask outside India ---
    mask = np.array([india_union.contains(Point(x, y))
                     for y, x in zip(grid_lat_mesh.ravel(), grid_lon_mesh.ravel())])
    mask = mask.reshape(grid_lat_mesh.shape)

    grid_poll_masked = np.where(mask, grid_poll, np.nan)

    # --- 5. Gradient field ---
    dlat = grid_lat[1] - grid_lat[0]
    dlon = grid_lon[1] - grid_lon[0]
    grad_lat, grad_lon = np.gradient(grid_poll, dlat, dlon)

    mag = np.sqrt(grad_lat**2 + grad_lon**2)
    u_unit = np.divide(grad_lon, mag, out=np.zeros_like(grad_lon), where=mag != 0)
    v_unit = np.divide(grad_lat, mag, out=np.zeros_like(grad_lat), where=mag != 0)

    # Mask arrows outside India
    u_unit_masked = np.where(mask, u_unit, np.nan)
    v_unit_masked = np.where(mask, v_unit, np.nan)

    # --- 6. Global normalization if vmin/vmax not given ---
    vals = np.log10(df.loc[:, f"{year}-01":f"{year}-12"].values.flatten() + 1e-12)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    # --- 7. Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Pollution field
    cf = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_poll_masked,
                     levels=15, cmap=cmap, alpha=0.9, zorder=1,
                     vmin=vmin, vmax=vmax)

    # Iso-lines (white for contrast)
    cs = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_poll_masked,
                    levels=12, colors="white", linewidths=0.7, zorder=2,
                    vmin=vmin, vmax=vmax)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors="white")

    # Gradient arrows
    ax.quiver(grid_lon_mesh[::step, ::step],
              grid_lat_mesh[::step, ::step],
              u_unit_masked[::step, ::step],
              v_unit_masked[::step, ::step],
              color="red", scale=30, width=0.004, alpha=0.7, zorder=3)

    # Bold India boundary
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=4)

    # Restrict to bounding box
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title(f"Pollution ({particle}) Gradient ({year}-{month:02d})", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.7)
    cbar.set_label("Pollution (log10 scaled, fixed across months)")

    plt.savefig(out_path)
    plt.close()

    summary1 = generate_inference(
        u_unit_masked, v_unit_masked,
        mag, mask,
        region_name="J&K + HP + Punjab + Uttarakhand + Haryana",
        grid_lon=grid_lon_mesh,
        grid_lat=grid_lat_mesh,
        grid_pollutant=grid_poll_masked,
        district_shapefile=district_file
    )
    return summary1

def animate_pollution_gradient_year(datafile, year, particle, out_gif, step=3, smooth=True,
                                    cmap="RdYlGn_r", smooth_frames=15):
    """
    Create a smooth GIF of monthly pollution gradient + store month-wise inferences.
    Returns: summaries dict
    """

    df = pd.read_csv(datafile)

    # --- Bounding box ---
    lat_min, lat_max = 29, 37
    lon_min, lon_max = 72.5, 81
    grid_lat = np.linspace(lat_min, lat_max, 200)
    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # --- Global vmin/vmax across year (safe log10) ---
    months_present = [c for c in df.columns if c.startswith(f"{year}-")]
    if not months_present:
        raise ValueError(f"No columns for year {year} found in {datafile}")

    vals = []
    for c in months_present:
        colvals = df[c].replace([np.inf, -np.inf], np.nan).dropna().values
        if colvals.size:
            vals.append(np.log10(colvals + 1e-12))
    if not vals:
        raise ValueError("No numeric pollution values found for requested year")
    all_vals = np.concatenate(vals)
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    # --- Precompute all available months ---
    monthly_data = []
    summaries = {}

    points = df[["lat", "lon"]].values

    for month in range(1, 13):
        colname = f"{year}-{month:02d}"
        if colname not in df.columns:
            continue

        raw = df[colname].replace([np.inf, -np.inf], np.nan).values
        if np.all(np.isnan(raw)):
            continue

        values = np.log10(np.nan_to_num(raw, nan=0.0) + 1e-12)

        # Interpolate onto grid
        grid_poll = griddata(points, values, (grid_lat_mesh, grid_lon_mesh), method="linear")
        if np.all(np.isnan(grid_poll)):
            grid_poll = griddata(points, values, (grid_lat_mesh, grid_lon_mesh), method="nearest")

        if smooth:
            nan_mask = np.isnan(grid_poll)
            grid_poll_filled = np.where(nan_mask, 0.0, grid_poll)
            smooth_arr = gaussian_filter(grid_poll_filled, sigma=2, mode="nearest")
            grid_poll = np.where(nan_mask, np.nan, smooth_arr)

        # Mask outside India
        pts_lon = grid_lon_mesh.ravel()
        pts_lat = grid_lat_mesh.ravel()
        mask = np.array([india_union.contains(Point(lon, lat)) for lon, lat in zip(pts_lon, pts_lat)])
        mask = mask.reshape(grid_lat_mesh.shape)
        grid_poll_masked = np.where(mask, grid_poll, np.nan)

        # Gradient field
        dlat = grid_lat[1] - grid_lat[0]
        dlon = grid_lon[1] - grid_lon[0]
        gp_tmp = np.nan_to_num(grid_poll, nan=0.0)
        grad_lat, grad_lon = np.gradient(gp_tmp, dlat, dlon)
        mag = np.sqrt(grad_lat**2 + grad_lon**2)
        u_unit = np.divide(grad_lon, mag, out=np.zeros_like(grad_lon), where=mag != 0)
        v_unit = np.divide(grad_lat, mag, out=np.zeros_like(grad_lat), where=mag != 0)
        u_unit_masked = np.where(mask, u_unit, np.nan)
        v_unit_masked = np.where(mask, v_unit, np.nan)

        monthly_data.append((month, grid_poll_masked, u_unit_masked, v_unit_masked))

    total_frames = (len(monthly_data) - 1) * smooth_frames
    if total_frames <= 0:
        raise ValueError("Computed zero total_frames; check monthly_data and smooth_frames")

    fig, ax = plt.subplots(figsize=(10, 10))
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=4)
    title = ax.set_title("", fontsize=14)

    _, init_poll, init_u, init_v = monthly_data[0]

    # Create ScalarMappable for continuous colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar

    im = ax.imshow(init_poll, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max),
                   cmap=cmap, norm=norm, interpolation='none', zorder=1)

    quiv = ax.quiver(grid_lon_mesh[::step, ::step], grid_lat_mesh[::step, ::step],
                     init_u[::step, ::step], init_v[::step, ::step],
                     scale=30, width=0.004, alpha=0.7, color='red', zorder=2)

    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Pollution (log10 scaled, fixed across months)")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    def update(frame):
        idx = frame // smooth_frames
        t = (frame % smooth_frames) / float(smooth_frames)

        if idx >= len(monthly_data) - 1:
            return [im, quiv, title]

        m1, poll1, u1, v1 = monthly_data[idx]
        m2, poll2, u2, v2 = monthly_data[idx + 1]

        poll_blend = (1 - t) * poll1 + t * poll2
        u_blend = (1 - t) * u1 + t * u2
        v_blend = (1 - t) * v1 + t * v2

        im.set_data(poll_blend)
        quiv.set_UVC(u_blend[::step, ::step], v_blend[::step, ::step])
        title.set_text(f"Pollution ({particle}) Gradient {year}-{m1:02d} → {m2:02d}")

        return [im, quiv, title]

    anim = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"✅ GIF saved as {out_gif}")
    inf=generate_infernces_pollution(out_gif)
    return inf