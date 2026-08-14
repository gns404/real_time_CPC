# Real-Time CPC Precipitation Outlook

This project produces period-based precipitation probability outlooks for the contiguous United States (CONUS) using CFSv2 forecasts and a Weather Regime Gaussian Mixture Model (WR-GMM).

All precipitation data and coefficients use the native CFSv2 Gaussian grid. The forecast ensemble consists of 12 CFSv2 initializations: 00, 06, 12, and 18 UTC on days 4, 5, and 6 of each month. Coefficients are applied separately to each initialization, and the resulting probabilities are averaged across the 12 members.

## Forecast periods

| Period | Definition |
|---|---|
| P1 | Day 16 through the end of the initialization month |
| P2 | The next calendar month |
| P3 | The second following calendar month |
| P4 | The third following calendar month |
| P5 | Three-month mean from P2 through P4 |

## File organization

### Training workflow: `Train_0`–`Train_6`

The file numbers indicate the order of the training workflow.

#### `Train_0_regrid_cpc_to_cfsv2.py`

Conservatively regrids daily CPC precipitation to the native Gaussian grid of a specified CFSv2 precipitation file. It preserves the latitude and longitude coordinates of the CFSv2 template instead of constructing a regular 1-degree grid.

Main outputs:

```text
cpc.198101-201012.cfsv2_native_grid.nc
cpc.201101-202208.cfsv2_native_grid.nc
```

Example:

```bash
python Train_0_regrid_cpc_to_cfsv2.py \
    --input /path/to/cpc.nc \
    --template CFSv2/precip_0506/prate.2012030506.nc \
    --output cpc.regridded.nc
```

#### `Train_1_GMM_train.py`

Trains a five-component Gaussian Mixture Model using daily CORe Z500 anomalies for 1981–2010.

Output:

```text
gmm_CORe.sav
```

#### `Train_2_make_precip_profile.py`

Creates monthly CPC precipitation profiles for each weather regime. A day is included in a profile when its GMM posterior probability for that regime is at least 0.9.

Outputs are written to `precip_profile/`:

```text
Precip01WT1.nc
...
Precip12WT5.nc
```

#### `Train_3_WR_WRGMM.py`

Applies the GMM to 2012–2021 CFSv2 Z500 forecasts and calculates daily weather-regime probabilities for each initialization. It then combines those probabilities with the monthly precipitation profiles to reconstruct WRGMM precipitation.

Initialization members:

```text
0400 0406 0412 0418
0500 0506 0512 0518
0600 0606 0612 0618
```

#### `Train_4_reconstruct_CORe_WRGMM_monthly.py`

Reconstructs historical monthly WRGMM precipitation for 1981–2010 using CORe Z500 anomalies and the monthly weather-regime precipitation profiles.

Output:

```text
CORe_WRGMM_monthly_precip.1981-2010.nc
```

#### `Train_5_calculate_coeff_by_member.py`

Processes the 12 CFSv2 initializations independently. For P1–P5, raw CFSv2 precipitation and WRGMM precipitation are blended using the squared historical CPC–WRGMM correlation. Coefficients are calculated after a fourth-root transformation.

Calculated coefficients:

- `clim`: forecast climatology by initialization month
- `sd`: forecast standard deviation by initialization month
- `r`: correlation between the forecast and CPC observations

Coefficients are saved separately under `coeff/<member>/`:

```text
coeff/0400/CFSv2.P1.exWRGMM.clim.120mon.nc
coeff/0400/CFSv2.P1.exWRGMM.sd.120mon.nc
coeff/0400/CFSv2.P1.exWRGMM.r.120mon.nc
```

Missing PRATE or WRGMM files are skipped. The number of samples used for each member and period is recorded in:

```text
coeff/coefficient_sample_count.txt
```

#### `Train_6_plot_clim_figures.py`

Creates diagnostic figures for the training-period weather-regime climatology:

- Monthly weather-regime frequency climatology
- Monthly precipitation profiles for each weather regime
- Z500 composites for each weather regime

Outputs are written to `PNG/train_climatology/`.

## Real-time workflow: `RealTime_*`

### `RealTime_CFSv2_forecast_download.sh`

Downloads the original PRATE and Z500 GRIB2 files for all 12 CFSv2 initializations in the current month. A year and month can also be supplied explicitly.

```bash
./RealTime_CFSv2_forecast_download.sh
./RealTime_CFSv2_forecast_download.sh 2025 08
```

Output layout:

```text
original_forecast/YYYYMM01/
├── tp/
│   └── prate.YYYYMMDDHH.grb2
└── z500/
    └── z500.YYYYMMDDHH.grb2
```

### `RealTime_ecmwf_forecast_download.py`

Downloads ECMWF seasonal total precipitation and Z500 data through the CDS API and stores them under `original_forecast/YYYYMM01/`. ECMWF data are not currently used by the CFSv2 P1–P5 outlook workflow.

### `RealTime_plot_P1_P5.py`

Calculates and plots P1–P5 weather-regime probabilities and precipitation tercile probabilities from the 12 CFSv2 initializations.

Processing steps:

1. Read the 12 CFSv2 Z500 GRIB2 forecasts and calculate daily means.
2. Remove the CORe Z500 climatology to calculate daily anomalies.
3. Calculate daily GMM posterior weather-regime probabilities.
4. Combine the probabilities with monthly precipitation profiles to reconstruct WRGMM precipitation.
5. Convert CFSv2 PRATE to `mm day-1` and calculate P1–P5 means.
6. Apply initialization-specific coefficients to calculate tercile probabilities.
7. Average the probabilities across the 12 initialization members.

Run the complete calculation and plotting workflow:

```bash
python RealTime_plot_P1_P5.py --init-date 20250801 --mode all
```

Run calculations only:

```bash
python RealTime_plot_P1_P5.py --init-date 20250801 --mode calculate
```

Regenerate figures from saved results:

```bash
python RealTime_plot_P1_P5.py --init-date 20250801 --mode plot
```

Main outputs:

```text
rt_output/YYYYMM01/CFSv2/
PNG/YYYYMM01/CFSv2_P1-P5_wt_probability_YYYYMM01.png
PNG/YYYYMM01/CFSv2_P1_..._tercile_outlook.png
...
PNG/YYYYMM01/CFSv2_P5_..._tercile_outlook.png
```

The `Issued` label on each figure represents the CFSv2 initialization period, days 4–6 of the corresponding month.

## Example workflow: `Example_*`

### `Example_CFSv2_forecast_download_202508.sh`

A fixed example that downloads the 12 CFSv2 initializations for August 2025.

```bash
nohup ./Example_CFSv2_forecast_download_202508.sh \
    > CFSv2_download_202508.log 2>&1 &
```

## Directory structure

```text
real_time_CPC/
├── Train_0_regrid_cpc_to_cfsv2.py
├── Train_1_GMM_train.py
├── Train_2_make_precip_profile.py
├── Train_3_WR_WRGMM.py
├── Train_4_reconstruct_CORe_WRGMM_monthly.py
├── Train_5_calculate_coeff_by_member.py
├── Train_6_plot_clim_figures.py
├── RealTime_CFSv2_forecast_download.sh
├── RealTime_ecmwf_forecast_download.py
├── RealTime_plot_P1_P5.py
├── Example_CFSv2_forecast_download_202508.sh
├── CFSv2/                  # Training CFSv2 data
├── coeff/                  # Member-specific P1–P5 coefficients
├── precip_profile/         # Monthly weather-regime precipitation profiles
├── original_forecast/      # Original real-time GRIB2 forecasts
├── rt_output/              # Intermediate and probability outputs
└── PNG/                    # Training and real-time figures
```

## Requirements

Main Python packages:

```text
numpy
pandas
xarray
scipy
scikit-learn
netCDF4
cfgrib
matplotlib
cartopy
shapely
cdsapi
```

External command-line tools:

```text
CDO
wget
```

## Data and path notes

- Large NetCDF, GRIB2, model, and generated figure files are excluded through `.gitignore`.
- Empty directory structures are preserved with `.gitkeep` files.
- Several scripts contain absolute paths under `/work05/home/jihun/real_time_CPC` and to external reanalysis data. Update the path variables near the top of each script when running the workflow on another system.
- `gmm_CORe.sav`, CPC data, CORe Z500 data, coefficients, and precipitation profiles must be provided separately because they are not stored in Git.
