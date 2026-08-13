from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
CFS_DIR = BASE_DIR / "CFSv2"
OUTPUT_DIR = BASE_DIR / "coeff"

MEMBERS = tuple(
    (day, hour) for day in (4, 5, 6) for hour in (0, 6, 12, 18)
)

CPC_PATH = BASE_DIR / "cpc.201101-202208.cfsv2_native_grid.nc"
CPC_HIST_PATH = BASE_DIR / "cpc.198101-201012.cfsv2_native_grid.nc"
WRGMM_HIST_PATH = BASE_DIR / "CORe_WRGMM_monthly_precip.1981-2010.nc"

def get_period(year, month, period):
    init_month = pd.Timestamp(year, month, 1)

    if period == 1:
        start = init_month + pd.DateOffset(days=15)
        end = init_month + pd.offsets.MonthEnd(0)
    elif period == 2:
        start = init_month + pd.DateOffset(months=1)
        end = start + pd.offsets.MonthEnd(0)
    elif period == 3:
        start = init_month + pd.DateOffset(months=2)
        end = start + pd.offsets.MonthEnd(0)
    elif period == 4:
        start = init_month + pd.DateOffset(months=3)
        end = start + pd.offsets.MonthEnd(0)
    elif period == 5:
        start = init_month + pd.DateOffset(months=1)
        end = init_month + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)

    return start, end

def fourth_root(data):
    return np.sign(data) * np.abs(data) ** 0.25

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

def save_coeff(data, path, varname, member, period, sample_count):
    data = data.rename(varname)
    data.attrs.update(
        member=member,
        period=f"P{period}",
        sample_count=sample_count,
        transformation="fourth root",
    )
    data.to_netcdf(
        path,
        encoding={varname: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    print(f"Saved: {path}")

def make_hist_weight():
    savepath = OUTPUT_DIR / "hist.r2.1981-2010.cfsv2_native.nc"

    if savepath.exists():
        hist_r2 = xr.open_dataset(savepath).hist_r2.load()
        rename = {}
        if "lat" in hist_r2.dims:
            rename["lat"] = "latitude"
        if "lon" in hist_r2.dims:
            rename["lon"] = "longitude"

        if rename:
            hist_r2 = hist_r2.rename(rename)
            temp_path = savepath.with_suffix(".tmp.nc")
            hist_r2.to_netcdf(temp_path)
            temp_path.replace(savepath)

        return hist_r2

    cpc_hist = xr.open_dataset(CPC_HIST_PATH).precip
    cpc_hist = cpc_hist.resample(time="MS").mean()

    wrgmm_hist = xr.open_dataset(WRGMM_HIST_PATH).precip_wrgmm
    wrgmm_hist = wrgmm_hist.rename(
        {"lat": "latitude", "lon": "longitude"}
    )
    cpc_hist, wrgmm_hist = xr.align(cpc_hist, wrgmm_hist, join="inner")

    hist_r2 = (xr.corr(cpc_hist, wrgmm_hist, dim="time") ** 2).rename("hist_r2")
    hist_r2.load().to_netcdf(savepath)
    print(f"Saved: {savepath}")
    return hist_r2
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument(
        "--members", nargs="+",
        default=[f"{day:02d}{hour:02d}" for day, hour in MEMBERS],
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpc_pr = xr.open_dataset(CPC_PATH).precip.load()
    hist_weight = make_hist_weight()
    sample_count_list = []

    # day/hour가 최외곽 반복문: 각 시점을 독립적으로 계산한다.
    for day, hour in MEMBERS:
        member = f"{day:02d}{hour:02d}"
        if member not in args.members:
            continue

        print(f"\n===== START {member} =====")

        CPC = {period: [] for period in range(1, 6)}
        CFSv2 = {period: [] for period in range(1, 6)}
        CFSv2_WRGMM = {period: [] for period in range(1, 6)}
        missing = []

        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                ym = f"{year}{month:02d}"
                output_time = pd.Timestamp(year, month, 1)

                cfsv2_path = (CFS_DIR / f"precip_{member}" / f"prate.{ym}{day:02d}{hour:02d}.nc")
                wrgmm_path = (CFS_DIR / f"z500_{member}" / "precip_WRGMM_CFSR" / f"precip.WRGMM.{ym}.nc")

                try:
                    temp_cfsv2 = xr.open_dataset(cfsv2_path).precip.load()
                    temp_wrgmm = xr.open_dataarray(wrgmm_path)
                    temp_wrgmm = temp_wrgmm.rename(
                        {"lat": "latitude", "lon": "longitude"}
                    ).load()
                except (FileNotFoundError, OSError) as error:
                    print(f"SKIP {ym} {member}: {error}")
                    missing.append([ym, str(error)])
                    continue

                for period in range(1, 6):
                    start, end = get_period(year, month, period)

                    temp_cpc = cpc_pr.sel(time=slice(start, end))
                    temp_cfs = temp_cfsv2.sel(time=slice(start, end))
                    temp_wrgmm_period = temp_wrgmm.sel(time=slice(start, end))

                    if min(
                        temp_cpc.time.size,
                        temp_cfs.time.size,
                        temp_wrgmm_period.time.size,
                    ) == 0:
                        print(f"NO DATA {ym} {member} P{period}")
                        continue

                    cpc_mean = temp_cpc.mean("time")
                    cfsv2_mean = temp_cfs.mean("time")
                    wrgmm_mean = temp_wrgmm_period.mean("time")

                    cfsv2_mean = xr.where(cpc_mean.isnull(), np.nan, cfsv2_mean)
                    wrgmm_mean = xr.where(cpc_mean.isnull(), np.nan, wrgmm_mean)

                    CPC[period].append(cpc_mean.expand_dims(time=[output_time]))
                    CFSv2[period].append(cfsv2_mean.expand_dims(time=[output_time]))
                    CFSv2_WRGMM[period].append(
                        wrgmm_mean.expand_dims(time=[output_time])
                    )

        member_output = OUTPUT_DIR / member
        member_output.mkdir(parents=True, exist_ok=True)

        for period in range(1, 6):
            if len(CFSv2[period]) == 0:
                print(f"NO VALID SAMPLE {member} P{period}")
                sample_count_list.append([member, f"P{period}", 0, 120])
                continue

            cpc = xr.concat(CPC[period], dim="time")
            cfsv2 = xr.concat(CFSv2[period], dim="time")
            wrgmm = xr.concat(CFSv2_WRGMM[period], dim="time")

            exwrgmm = cfsv2 * (1 - hist_weight) + wrgmm * hist_weight
            coeff = calculate_coefficients(exwrgmm, cpc)
            clim, sd, corr, obs_clim, obs_sd = coeff

            sample_count = len(CFSv2[period])
            sample_count_list.append([member, f"P{period}", sample_count, 120])
            prefix = member_output / f"CFSv2.P{period}.exWRGMM"

            save_coeff(clim, Path(f"{prefix}.clim.120mon.nc"),
                       "clim", member, period, sample_count)
            save_coeff(sd, Path(f"{prefix}.sd.120mon.nc"),
                       "sd", member, period, sample_count)
            save_coeff(corr, Path(f"{prefix}.r.120mon.nc"),
                       "r", member, period, sample_count)
            save_coeff(obs_clim, member_output / f"CPC.P{period}.pr.clim.120mon.nc",
                       "clim", member, period, sample_count)
            save_coeff(obs_sd, member_output / f"CPC.P{period}.pr.sd.120mon.nc",
                       "sd", member, period, sample_count)

        if missing:
            pd.DataFrame(missing, columns=["initialization_month", "reason"]).to_csv(
                member_output / "missing_files.csv", index=False
            )

        print(f"===== END {member} =====")

    count_path = OUTPUT_DIR / "coefficient_sample_count.txt"
    with open(count_path, "w") as file:
        file.write("member period used total\n")
        for member, period, used, total in sample_count_list:
            file.write(f"{member} {period} {used} {total}\n")

    print(f"Saved: {count_path}")


if __name__ == "__main__":
    main()
