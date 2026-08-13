"""Create monthly weather-type precipitation profiles on the CFSv2 1-degree grid."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
Z500_PATH = Path("/work05/home/jihun/seasonal_forecast/reanalysis/CORe.daily.z500.ano.1981-2010.nc")
PRECIP_PATH = BASE_DIR / "cpc.198101-201012.cfsv2_native_grid.nc"
GMM_PATH = BASE_DIR / "gmm_CORe.sav"
OUTPUT_DIR = BASE_DIR / "precip_profile"
N_WT = 5
PROBABILITY_THRESHOLD = 0.9


def main() -> None:
    with xr.open_dataset(Z500_PATH) as ds:
        z500 = ds["z500"].squeeze().load()

    nt, ny, nx = z500.shape
    z500_train = np.reshape(z500.values, (nt, ny * nx), order="F")
    with GMM_PATH.open("rb") as handle:
        gmm = pickle.load(handle)
    probabilities = gmm.predict_proba(z500_train)

    with xr.open_dataset(PRECIP_PATH) as ds:
        precip = ds["precip"].load()
        common_time = np.intersect1d(z500.time.values, precip.time.values)
        if common_time.size != nt:
            raise ValueError(
                f"Z500/CPC time mismatch: {nt} Z500 days but "
                f"{common_time.size} common days"
            )
        precip = precip.sel(time=z500.time.values)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        months = pd.DatetimeIndex(z500.time.values).month
        for month in range(1, 13):
            print(f"month: {month:02d}")
            for wt in range(N_WT):
                selected = ((months == month) &
                            (probabilities[:, wt] >= PROBABILITY_THRESHOLD))
                count = int(selected.sum())
                if count == 0:
                    raise ValueError(
                        f"No dates selected for month={month:02d}, WT={wt + 1}"
                    )

                profile = precip.isel(time=np.flatnonzero(selected)).mean(
                    "time", skipna=True
                )
                profile.name = "precip"
                profile.attrs.update(
                    weather_type=wt + 1,
                    month=month,
                    probability_threshold=PROBABILITY_THRESHOLD,
                    sample_count=count,
                    source_file=str(PRECIP_PATH),
                )
                output = OUTPUT_DIR / f"Precip{month:02d}WT{wt + 1}.nc"
                temporary = output.with_suffix(".tmp.nc")
                profile.to_netcdf(
                    temporary,
                    encoding={"precip": {"zlib": True, "complevel": 4}},
                )
                temporary.replace(output)
                print(f"  WT{wt + 1}: {count} days -> {output.name}")


if __name__ == "__main__":
    main()
