"""Plot CORe weather-regime training climatology figures."""

from pathlib import Path
import argparse
import calendar
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader


BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
PNG_DIR = BASE_DIR / "PNG" / "train_climatology"
PROFILE_DIR = BASE_DIR / "precip_profile"
Z500_PATH = Path(
    "/work05/home/jihun/seasonal_forecast/reanalysis/"
    "CORe.daily.z500.ano.1981-2010.nc"
)
GMM_PATH = BASE_DIR / "gmm_CORe.sav"
STATE_SHP = BASE_DIR / "cb_2018_us_state_500k" / "cb_2018_us_state_500k.shp"

N_WR = 5
WR_NAMES = [
    "Pacific Trough",
    "Alaskan Trough",
    "Alaskan Ridge",
    "Pacific Ridge",
    "Pacific Wavetrain",
]
WR_SHORT = ["PT", "AT", "AR", "PR", "PW"]
WR_COLORS = ["#2ab07f", "#f04b0b", "#8173ad", "#e3b332", "#b85d1c"]


def load_z500_anomaly():
    with xr.open_dataset(Z500_PATH) as ds:
        return ds["z500"].squeeze(drop=True).load()


def load_gmm():
    with GMM_PATH.open("rb") as handle:
        return pickle.load(handle)


def flatten_z500(z500):
    nt, ny, nx = z500.shape
    return z500.values.reshape(nt, ny * nx, order="F"), ny, nx


def predict_wr(z500, gmm):
    z500_flat, _, _ = flatten_z500(z500)
    probability = gmm.predict_proba(z500_flat)
    assigned = probability.argmax(axis=1) + 1
    return probability, assigned


def get_states_feature():
    reader = shapereader.Reader(STATE_SHP)
    return cfeature.ShapelyFeature(
        list(reader.geometries()), ccrs.PlateCarree()
    )


def add_map_base(ax, extent, states, labels=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(states, facecolor="none", linewidth=0.45, edgecolor="0.25")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.45)
    gl = ax.gridlines(
        draw_labels=labels,
        xlocs=np.arange(-180, 181, 20),
        ylocs=np.arange(-90, 91, 10),
        linestyle=":",
        linewidth=0.35,
        color="0.45",
    )
    gl.top_labels = False
    gl.right_labels = False
    if labels:
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}


def compute_monthly_frequency(z500, assigned):
    frame = pd.DataFrame(
        {"month": pd.DatetimeIndex(z500.time.values).month, "wr": assigned}
    )
    counts = (
        frame.groupby(["month", "wr"])
        .size()
        .unstack("wr", fill_value=0)
        .reindex(index=range(1, 13), columns=range(1, N_WR + 1), fill_value=0)
    )
    frequency = counts.div(counts.sum(axis=1), axis=0) * 100.0
    frequency.index = [calendar.month_abbr[m] for m in frequency.index]
    frequency.columns = WR_NAMES
    return frequency


def plot_wr_frequency(frequency):
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    x = np.arange(12)
    bottom = np.zeros(12)
    for wr, name in enumerate(WR_NAMES):
        values = frequency[name].to_numpy()
        ax.bar(
            x, values, bottom=bottom, width=0.78,
            color=WR_COLORS[wr], edgecolor="black", linewidth=0.6, label=name,
        )
        for xpos, value, base in zip(x, values, bottom):
            if value >= 8.0:
                ax.text(
                    xpos, base + value / 2, f"{value:.0f}",
                    ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold",
                )
        bottom += values

    ax.set_xticks(x, frequency.index)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Frequency (%)")
    ax.set_title(
        "Monthly Weather-Regime Frequency Climatology (CORe 1981–2010)",
        fontsize=15, fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=N_WR, frameon=False,
    )
    output = PNG_DIR / "CORe_monthly_WR_frequency_climatology.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    frequency.to_csv(
        PNG_DIR / "CORe_monthly_WR_frequency_climatology.csv",
        float_format="%.3f",
    )
    print(f"Saved: {output}")


def load_precip_profiles():
    profiles = []
    missing = []
    for wr in range(1, N_WR + 1):
        monthly = []
        for month in range(1, 13):
            path = PROFILE_DIR / f"Precip{month:02d}WT{wr}.nc"
            if not path.exists():
                missing.append(path.name)
                continue
            with xr.open_dataset(path) as ds:
                variable = "precip" if "precip" in ds else next(iter(ds.data_vars))
                field = ds[variable].load().expand_dims(month=[month])
            monthly.append(field)
        if len(monthly) == 12:
            profiles.append(xr.concat(monthly, dim="month").expand_dims(wr=[wr]))
    if missing:
        raise FileNotFoundError("Missing precipitation profiles: " + ", ".join(missing))
    return xr.concat(profiles, dim="wr")


def prepare_precip_lon(profiles):
    longitude = ((profiles.longitude + 180) % 360) - 180
    return profiles.assign_coords(longitude=longitude).sortby("longitude")


def common_precip_levels(profiles):
    upper = float(profiles.quantile(0.99, skipna=True))
    upper = max(1.0, np.ceil(upper * 2.0) / 2.0)
    return np.linspace(0.0, upper, 17)


def plot_monthly_precip_profiles(profiles):
    profiles = prepare_precip_lon(profiles)
    levels = common_precip_levels(profiles)
    states = get_states_feature()

    for wr in range(1, N_WR + 1):
        fig, axes = plt.subplots(
            3, 4, figsize=(15.5, 9.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        for month, ax in enumerate(axes.flat, start=1):
            field = profiles.sel(wr=wr, month=month)
            plot = ax.contourf(
                field.longitude, field.latitude, field,
                levels=levels, cmap="Blues", extend="max",
                transform=ccrs.PlateCarree(),
            )
            add_map_base(ax, [-130, -65, 20, 50], states)
            # sample_count = field.attrs.get("sample_count")
            # suffix = f" (n={sample_count})" if sample_count is not None else ""
            # ax.set_title(f"{calendar.month_abbr[month]}{suffix}", fontsize=11)
            ax.set_title(f"{calendar.month_abbr[month]}", fontsize=11)

        colorbar = fig.colorbar(
            plot, ax=axes, orientation="horizontal",
            shrink=0.72, pad=0.035, fraction=0.035, aspect=45,
        )
        units = profiles.attrs.get("units", "mm day$^{-1}$")
        colorbar.set_label(f"Mean precipitation ({units})")
        fig.suptitle(
            f"{WR_NAMES[wr - 1]}: Monthly Precipitation Profiles (1981–2010)",
            fontsize=17, fontweight="bold",
        )
        output = PNG_DIR / f"CPC_monthly_precip_profile_WR{wr}_{WR_SHORT[wr - 1]}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output}")


def plot_wr_composites(z500, gmm):
    _, ny, nx = flatten_z500(z500)
    means = gmm.means_.reshape(N_WR, ny, nx, order="F")
    states = get_states_feature()
    levels = np.linspace(-100, 100, 41)
    fig, axes = plt.subplots(
        1, N_WR, figsize=(18, 4.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    for wr, ax in enumerate(axes, start=1):
        plot = ax.contourf(
            z500.longitude, z500.latitude, means[wr - 1],
            levels=levels, cmap="RdBu_r", extend="both",
            transform=ccrs.PlateCarree(),
        )
        add_map_base(ax, [-140, -60, 20, 60], states)
        ax.set_title(f"WR{wr}: {WR_NAMES[wr - 1]}", fontsize=11, fontweight="bold")

    colorbar = fig.colorbar(
        plot, ax=axes, orientation="horizontal",
        shrink=0.72, pad=0.035, fraction=0.035, aspect=45,
    )
    colorbar.set_label("Z500 anomaly (m)")
    fig.suptitle(
        "CORe Weather-Regime Z500 Composites (1981–2010)",
        fontsize=17, fontweight="bold",
    )
    output = PNG_DIR / "CORe_WR_Z500_composites.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["all", "frequency", "precip", "composite"],
        default="all", help="Generate only one figure group.",
    )
    args = parser.parse_args()
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    z500 = load_z500_anomaly()
    gmm = load_gmm()
    _, assigned = predict_wr(z500, gmm)

    if args.only in ("all", "frequency"):
        plot_wr_frequency(compute_monthly_frequency(z500, assigned))
    if args.only in ("all", "precip"):
        plot_monthly_precip_profiles(load_precip_profiles())
    if args.only in ("all", "composite"):
        plot_wr_composites(z500, gmm)


if __name__ == "__main__":
    main()
