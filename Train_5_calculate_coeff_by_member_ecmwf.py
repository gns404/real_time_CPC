from pathlib import Path
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
ECMWF_DIR = Path("/work05/home/jihun/seasonal_forecast/ECMWF")
Z500_DIR = ECMWF_DIR / "z500_ens25"
PRECIP_DIR = ECMWF_DIR / "precip_ens25"
OUTPUT_DIR = BASE_DIR / "coeff" / "ECMWF"

CPC_PATH = BASE_DIR / "cpc.201101-202208.cfsv2_native_grid.nc"
CPC_HIST_PATH = BASE_DIR / "cpc.198101-201012.cfsv2_native_grid.nc"
WRGMM_HIST_PATH = BASE_DIR / "CORe_WRGMM_monthly_precip.1981-2010.nc"
GMM_PATH = BASE_DIR / "gmm_CORe.sav"
PROFILE_DIR = BASE_DIR / "precip_profile"

N_WR = 5
PERIODS = range(1, 3)


def get_period(year, month, period):
    init_month = pd.Timestamp(year, month, 1)

    if period == 1:
        start = init_month + pd.DateOffset(months=1)
        end = start + pd.offsets.MonthEnd(0)
    elif period == 2:
        start = init_month + pd.DateOffset(months=1)
        end = init_month + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)

    return start, end


def fourth_root(data):
    return np.sign(data) * np.abs(data) ** 0.25


def open_first_dataarray(path):
    with xr.open_dataset(path) as ds:
        return ds[next(iter(ds.data_vars))].load()


def load_profiles():
    profiles = []

    for month in range(1, 13):
        monthly = []
        for wr in range(1, N_WR + 1):
            path = PROFILE_DIR / f"Precip{month:02d}WT{wr}.nc"
            with xr.open_dataset(path) as ds:
                profile = ds["precip"].load()
            monthly.append(profile.expand_dims(wr=[wr]))

        monthly = xr.concat(monthly, dim="wr")
        profiles.append(monthly.expand_dims(month=[month]))

    return xr.concat(profiles, dim="month").transpose(
        "month", "wr", "latitude", "longitude"
    )


def make_hist_weight():
    savepath = BASE_DIR / "coeff" / "hist.r2.1981-2010.cfsv2_native.nc"

    if savepath.exists():
        with xr.open_dataset(savepath) as ds:
            return ds["hist_r2"].load()

    with xr.open_dataset(CPC_HIST_PATH) as ds:
        cpc_hist = ds["precip"].resample(time="MS").mean().load()

    with xr.open_dataset(WRGMM_HIST_PATH) as ds:
        wrgmm_hist = ds["precip_wrgmm"].rename(
            {"lat": "latitude", "lon": "longitude"}
        ).load()

    cpc_hist, wrgmm_hist = xr.align(cpc_hist, wrgmm_hist, join="inner")
    hist_r2 = (xr.corr(cpc_hist, wrgmm_hist, dim="time") ** 2).rename("hist_r2")
    hist_r2.to_netcdf(savepath)
    print(f"Saved: {savepath}")
    return hist_r2


def load_ecmwf_precip(path, target_lat, target_lon):
    """Read ECMWF precipitation and regrid it in memory only."""
    with xr.open_dataset(path) as ds:
        precip = ds[next(iter(ds.data_vars))]

        # ECMWF precipitation is on the 0.25-degree CPC grid.  Interpolate to
        # the exact CFSv2-native coordinates used by the regridded CPC data.
        precip = precip.interp(
            lat=target_lat.values,
            lon=target_lon.values,
            method="linear",
        )
        precip = precip.rename({"lat": "latitude", "lon": "longitude"})
        precip = precip.assign_coords(
            latitude=target_lat.values,
            longitude=target_lon.values,
        )
        precip = precip.clip(min=0).load()

    precip.name = "precip_raw"
    return precip.transpose("ensemble", "time", "latitude", "longitude")


def load_ecmwf_wrgmm(path, gmm, profiles):
    with xr.open_dataset(path) as ds:
        z500 = ds[next(iter(ds.data_vars))].squeeze("plev", drop=True).load()

    z500 = z500.transpose("ensemble", "time", "latitude", "longitude")
    ne, nt, ny, nx = z500.shape
    predictors = z500.values.reshape(ne, nt, ny * nx, order="F")
    predictors = predictors.reshape(ne * nt, ny * nx, order="C")

    if not np.isfinite(predictors).all():
        raise ValueError(f"Non-finite Z500 values in {path}")
    if gmm.n_features_in_ != predictors.shape[1]:
        raise ValueError(
            f"GMM expects {gmm.n_features_in_} features, "
            f"but ECMWF Z500 has {predictors.shape[1]}"
        )

    probability = gmm.predict_proba(predictors).reshape(ne, nt, N_WR)
    probability = xr.DataArray(
        probability,
        coords={
            "ensemble": z500.ensemble.values,
            "time": z500.time.values,
            "wr": np.arange(1, N_WR + 1),
        },
        dims=("ensemble", "time", "wr"),
    )

    daily_profiles = profiles.sel(
        month=xr.DataArray(z500.time.dt.month.values, dims="time")
    ).drop_vars("month")
    daily_profiles = daily_profiles.assign_coords(time=z500.time.values)
    wrgmm = xr.dot(probability, daily_profiles, dims="wr")
    wrgmm.name = "precip_wrgmm"
    return wrgmm.transpose("ensemble", "time", "latitude", "longitude")


def calculate_coefficients(forecast, observation):
    forecast = fourth_root(forecast)
    observation = fourth_root(observation)

    clim = forecast.groupby("time.month").mean("time")
    sd = forecast.groupby("time.month").std("time", ddof=1)
    obs_clim = observation.groupby("time.month").mean("time")
    obs_sd = observation.groupby("time.month").std("time", ddof=1)

    corr = []
    for month in clim.month.values:
        temp_fcst = forecast.where(forecast.time.dt.month == month, drop=True)
        temp_obs = observation.where(observation.time.dt.month == month, drop=True)
        temp_r = xr.corr(temp_fcst, temp_obs, dim="time")
        corr.append(temp_r.expand_dims(month=[month]))

    corr = xr.concat(corr, dim="month").clip(-0.9999, 0.9999)
    return clim, sd, corr, obs_clim, obs_sd


def save_coeff(data, path, varname, period, sample_count):
    data = data.rename(varname)
    data.attrs.update(
        model="ECMWF",
        ensemble_members=data.sizes.get("ensemble", 1),
        period=f"P{period}",
        sample_count=sample_count,
        transformation="fourth root",
        precipitation_regrid="linear interpolation to CPC CFSv2-native grid",
    )
    data.to_netcdf(
        path,
        encoding={varname: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(CPC_PATH) as ds:
        cpc_pr = ds["precip"].load()

    target_lat = cpc_pr.latitude
    target_lon = cpc_pr.longitude
    hist_weight = make_hist_weight()
    profiles = load_profiles()

    with GMM_PATH.open("rb") as file:
        gmm = pickle.load(file)

    CPC = {period: [] for period in PERIODS}
    ECMWF = {period: [] for period in PERIODS}
    ECMWF_WRGMM = {period: [] for period in PERIODS}
    missing = []

    for year in range(args.start_year, args.end_year + 1):
        for month in args.months:
            init_date = f"{year}{month:02d}01"
            precip_path = PRECIP_DIR / f"ECMWF_{init_date}.nc"
            z500_path = Z500_DIR / f"ECMWF_{init_date}.nc"
            print(f"\n===== {init_date} =====")

            try:
                precip = load_ecmwf_precip(precip_path, target_lat, target_lon)
                wrgmm = load_ecmwf_wrgmm(z500_path, gmm, profiles)
                precip, wrgmm = xr.align(precip, wrgmm, join="inner")
            except (FileNotFoundError, OSError, ValueError) as error:
                print(f"SKIP {init_date}: {error}")
                missing.append([init_date, str(error)])
                continue

            output_time = pd.Timestamp(year, month, 1)

            for period in PERIODS:
                start, end = get_period(year, month, period)
                temp_cpc = cpc_pr.sel(time=slice(start, end))
                temp_ecmwf = precip.sel(time=slice(start, end))
                temp_wrgmm = wrgmm.sel(time=slice(start, end))

                if min(
                    temp_cpc.time.size,
                    temp_ecmwf.time.size,
                    temp_wrgmm.time.size,
                ) == 0:
                    print(f"NO DATA {init_date} P{period}")
                    continue

                cpc_mean = temp_cpc.mean("time")
                ecmwf_mean = temp_ecmwf.mean("time")
                wrgmm_mean = temp_wrgmm.mean("time")

                ecmwf_mean = xr.where(cpc_mean.notnull(), ecmwf_mean, np.nan)
                wrgmm_mean = xr.where(cpc_mean.notnull(), wrgmm_mean, np.nan)

                CPC[period].append(cpc_mean.expand_dims(time=[output_time]))
                ECMWF[period].append(ecmwf_mean.expand_dims(time=[output_time]))
                ECMWF_WRGMM[period].append(wrgmm_mean.expand_dims(time=[output_time]))

    sample_count_list = []
    expected_count = (args.end_year - args.start_year + 1) * len(args.months)

    for period in PERIODS:
        sample_count = len(ECMWF[period])
        sample_count_list.append([f"P{period}", sample_count, expected_count])

        if sample_count == 0:
            print(f"NO VALID SAMPLE P{period}")
            continue

        cpc = xr.concat(CPC[period], dim="time")
        ecmwf = xr.concat(ECMWF[period], dim="time")
        wrgmm = xr.concat(ECMWF_WRGMM[period], dim="time")

        exwrgmm = ecmwf * (1 - hist_weight) + wrgmm * hist_weight
        clim, sd, corr, obs_clim, obs_sd  = calculate_coefficients(exwrgmm, cpc)

        prefix = args.output_dir / f"ECMWF.P{period}.exWRGMM"
        save_coeff(clim, Path(f"{prefix}.clim.120mon.nc"),
                   "clim", period, sample_count)
        save_coeff(sd, Path(f"{prefix}.sd.120mon.nc"),
                   "sd", period, sample_count)
        save_coeff(corr, Path(f"{prefix}.r.120mon.nc"),
                   "r", period, sample_count)
        save_coeff(obs_clim, OUTPUT_DIR / f"CPC.P{period}.pr.clim.120mon.nc",
                   "clim", period, sample_count)
        save_coeff(obs_sd, OUTPUT_DIR / f"CPC.P{period}.pr.sd.120mon.nc",
                   "sd", period, sample_count)

    count_path = args.output_dir / "coefficient_sample_count.txt"
    with count_path.open("w") as file:
        file.write("period used total\n")
        for period, used, total in sample_count_list:
            file.write(f"{period} {used} {total}\n")
    print(f"Saved: {count_path}")

    if missing:
        missing_path = args.output_dir / "missing_files.csv"
        pd.DataFrame(missing, columns=["initialization", "reason"]).to_csv(
            missing_path, index=False
        )
        print(f"Saved: {missing_path}")


if __name__ == "__main__":
    main()
