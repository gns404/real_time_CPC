"""Reconstruct 1981-2010 monthly precipitation from CORe weather regimes.

Daily CORe Z500 anomalies are converted to five GMM posterior probabilities.
For each calendar month, the mean WR probabilities are multiplied by that
month's five precipitation profiles. The resulting monthly precipitation is
identical to reconstructing every day first and then taking the monthly mean,
because a calendar month's profile fields are constant within that month.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
DEFAULT_Z500 = Path(
    "/work05/home/jihun/seasonal_forecast/reanalysis/"
    "CORe.daily.z500.ano.1981-2010.nc"
)
DEFAULT_GMM = BASE_DIR / "gmm_CORe.sav"
DEFAULT_PROFILE_DIR = BASE_DIR / "precip_profile"
DEFAULT_OUTPUT = BASE_DIR / "CORe_WRGMM_monthly_precip.1981-2010.nc"
N_WR = 5


def load_z500(path: Path) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if "z500" not in ds:
            raise KeyError(f"'z500' is not present in {path}")
        z500 = ds["z500"].squeeze(drop=True).load()
    required = {"time", "latitude", "longitude"}
    if not required.issubset(z500.dims):
        raise ValueError(f"Expected Z500 dimensions {required}; got {z500.dims}")
    return z500.transpose("time", "latitude", "longitude")


def load_gmm(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def calculate_daily_wr_probability(z500: xr.DataArray, gmm) -> xr.DataArray:
    nt, ny, nx = z500.shape
    predictors = z500.values.reshape(nt, ny * nx, order="F")
    if not np.isfinite(predictors).all():
        count = int((~np.isfinite(predictors)).sum())
        raise ValueError(f"Z500 contains {count} non-finite predictor values")
    if getattr(gmm, "n_features_in_", predictors.shape[1]) != predictors.shape[1]:
        raise ValueError(
            f"GMM expects {gmm.n_features_in_} features, but Z500 has "
            f"{predictors.shape[1]}"
        )
    probability = gmm.predict_proba(predictors).astype("float32")
    return xr.DataArray(
        probability,
        coords={"time": z500.time.values, "wr": np.arange(1, N_WR + 1)},
        dims=("time", "wr"),
        name="wr_probability_daily",
        attrs={"description": "Daily GMM posterior weather-regime probability"},
    )


def load_profiles(profile_dir: Path) -> xr.DataArray:
    monthly_profiles = []
    reference_lat = None
    reference_lon = None
    for month in range(1, 13):
        regimes = []
        for wr in range(1, N_WR + 1):
            path = profile_dir / f"Precip{month:02d}WT{wr}.nc"
            if not path.is_file():
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                if "precip" not in ds:
                    raise KeyError(f"'precip' is not present in {path}")
                field = ds["precip"].load()
            rename = {}
            if "latitude" in field.dims:
                rename["latitude"] = "lat"
            if "longitude" in field.dims:
                rename["longitude"] = "lon"
            field = field.rename(rename).transpose("lat", "lon")
            if reference_lat is None:
                reference_lat = field.lat.values.copy()
                reference_lon = field.lon.values.copy()
            elif not (
                np.array_equal(field.lat.values, reference_lat)
                and np.array_equal(field.lon.values, reference_lon)
            ):
                raise ValueError(f"Precipitation-profile grid mismatch: {path}")
            regimes.append(field.expand_dims(wr=[wr]))
        monthly_profiles.append(
            xr.concat(regimes, dim="wr").expand_dims(month=[month])
        )
    profiles = xr.concat(monthly_profiles, dim="month").transpose(
        "month", "wr", "lat", "lon"
    )
    profiles.name = "precip_profile"
    return profiles


def reconstruct_monthly(
    daily_probability: xr.DataArray, profiles: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    monthly_probability = daily_probability.resample(time="MS").mean("time")
    monthly_probability.name = "wr_probability"
    monthly_probability.attrs.update(
        description="Monthly mean daily GMM posterior weather-regime probability"
    )

    fields = []
    for month in range(1, 13):
        selected = monthly_probability.where(
            monthly_probability.time.dt.month == month, drop=True
        )
        reconstructed = xr.dot(selected, profiles.sel(month=month), dims="wr")
        fields.append(reconstructed)
    monthly_precip = xr.concat(fields, dim="time").sortby("time")
    monthly_precip.name = "precip_wrgmm"
    monthly_precip.attrs.update(
        units="mm day-1",
        description="Monthly mean precipitation reconstructed from WR probabilities",
        reconstruction="sum over WR of monthly-mean probability times monthly WR profile",
    )

    dominant_wr = (monthly_probability.argmax("wr") + 1).astype("int8")
    dominant_wr.name = "dominant_wr"
    dominant_wr.attrs.update(
        description="Weather regime with maximum monthly-mean posterior probability",
        valid_values="1, 2, 3, 4, 5",
    )
    return monthly_precip, monthly_probability, dominant_wr


def save_output(
    output_path: Path,
    precip: xr.DataArray,
    probability: xr.DataArray,
    dominant_wr: xr.DataArray,
    z500_path: Path,
    gmm_path: Path,
    profile_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        {
            "precip_wrgmm": precip,
            "wr_probability": probability,
            "dominant_wr": dominant_wr,
        },
        attrs={
            "source_z500": str(z500_path),
            "gmm_model": str(gmm_path),
            "precipitation_profiles": str(profile_dir),
            "period": "1981-2010",
            "temporal_resolution": "monthly mean",
            "weather_regimes": N_WR,
        },
    )
    temporary = output_path.with_suffix(".tmp.nc")
    dataset.to_netcdf(
        temporary,
        encoding={
            "precip_wrgmm": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "wr_probability": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "dominant_wr": {"zlib": True, "complevel": 4, "dtype": "int8"},
        },
    )
    temporary.replace(output_path)
    print(f"Saved: {output_path}")
    print(
        f"Output: {precip.sizes['time']} months, "
        f"{precip.sizes['lat']} latitude x {precip.sizes['lon']} longitude"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z500", type=Path, default=DEFAULT_Z500)
    parser.add_argument("--gmm", type=Path, default=DEFAULT_GMM)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    z500 = load_z500(args.z500)
    gmm = load_gmm(args.gmm)
    profiles = load_profiles(args.profile_dir)
    daily_probability = calculate_daily_wr_probability(z500, gmm)
    precip, monthly_probability, dominant_wr = reconstruct_monthly(
        daily_probability, profiles
    )
    save_output(
        args.output, precip, monthly_probability, dominant_wr,
        args.z500, args.gmm, args.profile_dir,
    )


if __name__ == "__main__":
    main()
