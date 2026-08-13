from pathlib import Path
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import box

GRAVITY = 9.80665
N_WT = 5
PERIODS = (1, 5)
N_LEADS = 4  # retained only for unused monthly-WR plotting helpers below

BASE_DIR = Path('/work05/home/jihun/real_time_CPC')
RAW_DIR = BASE_DIR / 'original_forecast'
COEFF_DIR = BASE_DIR / 'coeff'
PROFILE_DIR = BASE_DIR / 'precip_profile'
OUTPUT_DIR = BASE_DIR / 'rt_output'

GMM_PATH = BASE_DIR / 'gmm_CORe.sav'
HIST_R_PATH = COEFF_DIR / 'hist.r2.1981-2010.cfsv2_native.nc'
CLIM_PATH = COEFF_DIR / 'CORe.daily.z500.clim.1981-2010.nc'
OBS_CLIM_PATH = COEFF_DIR / '0400/CPC.P1.pr.clim.120mon.nc'
# OBS_SD_PATH = COEFF_DIR / '0400/CPC.P1.pr.sd.120mon.nc'

MODEL_NAME = 'CFSv2'
CFSV2_MEMBERS = [f'{day:02d}{hour:02d}' for day in (4, 5, 6) for hour in (0, 6, 12, 18)]

# WR_LABELS = {1: 'Pacific Trough',2: 'Pacific Ridge',3: 'Pacific Wave',4: 'Alaskan Trough',5: 'Alaskan Ridge',}
WR_LABELS = {1: 'Pacific Trough',2: 'Alaskan Trough',3: 'Alaskan Ridge',4: 'Pacific Ridge',5: 'Pacific Wavetrain',}
WR_COLORS = {1: '#2ab07f',2: '#f04b0b',3: '#8173ad',4: '#e3b332',5: '#b85d1c',}
# WR_COLORS = {1: '#2ab07f',2: '#f04b0b',3: '#8173ad',4: '#e3b332',5: '#b85d1c',}

now = datetime.now()
year = now.year
month = now.month

year = 2025
month = 8

init_date = f"{year}{month:02d}01"
init_date

def open_dataarray(path, var_name=None):
    ds = xr.open_dataset(path)
    if var_name and var_name in ds.data_vars:
        return ds[var_name]
    return ds[next(iter(ds.data_vars))]

def load_gmm():
    with open(GMM_PATH, 'rb') as handle:
        return pickle.load(handle)

def load_core_climatology():
    return open_dataarray(CLIM_PATH, 'z500').squeeze('plev', drop=True)

def load_hist_r_weight():
    return open_dataarray(HIST_R_PATH, 'hist_r2')

def load_model_coeff(member, period, kind):
    path = COEFF_DIR / member / f'CFSv2.P{period}.exWRGMM.{kind}.120mon.nc'
    return open_dataarray(path, kind)

def load_precip_profile(month, wt):
    path = PROFILE_DIR / f'Precip{month:02d}WT{wt}.nc'
    return open_dataarray(path, 'precip')

def get_period(init_date, period):
    init_month = pd.Timestamp(init_date)
    if period == 1:
        return init_month + pd.DateOffset(days=15), init_month + pd.offsets.MonthEnd(0)
    if period == 5:
        start = init_month + pd.DateOffset(months=1)
        end = init_month + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)
        return start, end
    raise ValueError(period)

def open_grib_dataarray(path, field):
    ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
    candidates = ('prate', 'tp') if field == 'tp' else ('gh', 'z')
    for name in candidates:
        if name in ds:
            return ds[name]
    return ds[next(iter(ds.data_vars))]

def fourth_root(pr):
    return np.sign(pr) * np.abs(pr) ** 0.25

def to_minus180(lon):
    return ((lon + 180) % 360) - 180

def load_cfsv2_z500(init_date):
    members = []
    for member in CFSV2_MEMBERS:
        path = RAW_DIR / init_date / 'z500' / f'z500.{init_date[:6]}{member}.grb2'
        da = open_grib_dataarray(path, 'z500')
        valid_time = pd.to_datetime(da.valid_time.values)
        da = da.rename({
            'latitude': 'lat',
            'longitude': 'lon',
            'step': 'forecast_step',
        })
        da = da.drop_vars(['time', 'valid_time'], errors='ignore')
        da = da.assign_coords(time=('forecast_step', valid_time))
        da = da.swap_dims({'forecast_step': 'time'}).drop_vars('forecast_step')
        da = da.assign_coords(lon=to_minus180(da.lon)).sortby('lon').sortby('lat')
        da = da.sel(lat=slice(-5, 65), lon=slice(-150, -50))
        da = da.resample(time='1D').mean()
        members.append(da.expand_dims(ensemble=[member]))
    return xr.concat(members, dim='ensemble')

def load_cfsv2_tp(init_date):
    members = []
    for member in CFSV2_MEMBERS:
        path = RAW_DIR / init_date / 'tp' / f'prate.{init_date[:6]}{member}.grb2'
        da = open_grib_dataarray(path, 'tp')
        valid_time = pd.to_datetime(da.valid_time.values)
        da = da.rename({
            'latitude': 'lat',
            'longitude': 'lon',
            'step': 'forecast_step',
        })
        da = da.drop_vars(['time', 'valid_time'], errors='ignore')
        da = da.assign_coords(time=('forecast_step', valid_time))
        da = da.swap_dims({'forecast_step': 'time'}).drop_vars('forecast_step')
        da = da.sortby('lon').sortby('lat').sel(lat=slice(19, 51), lon=slice(228, 307))
        da = (da * 86400.0).resample(time='1D').mean()
        da.name = 'precip'
        members.append(da.expand_dims(ensemble=[member]))
    return xr.concat(members, dim='ensemble')

def regrid_to_core_grid(z500_daily, core_clim):
    z500_daily = z500_daily.interp(lon=core_clim.longitude, lat=core_clim.latitude, method='linear')
    drop_coords = [name for name in ['longitude', 'latitude'] if name in z500_daily.coords]
    if drop_coords:
        z500_daily = z500_daily.drop_vars(drop_coords)
    z500_daily = z500_daily.rename(lon='longitude', lat='latitude')
    return z500_daily.transpose('ensemble', 'time', 'latitude', 'longitude')

def regrid_tp_to_obs_grid(tp_daily, obs_grid):
    return tp_daily.rename(lat='latitude', lon='longitude').transpose(
        'ensemble', 'time', 'latitude', 'longitude'
    )

def make_daily_anomaly(z500_daily_rg, core_clim):
    day_index = z500_daily_rg.time.dt.dayofyear.values.astype(int) - 1
    clim_match = core_clim.isel(time=day_index).assign_coords(time=z500_daily_rg.time.values)
    anomaly = z500_daily_rg - clim_match
    anomaly.name = 'z500_anomaly'
    return anomaly

def predict_probabilities(z500_anomaly, gmm):
    data_r = z500_anomaly.values
    data_r = np.nan_to_num(data_r, nan=0.0)
    ne, nt, ny, nx = data_r.shape
    data_1 = data_r.reshape(ne, nt, ny * nx, order='F')
    data_f = data_1.reshape(ne * nt, ny * nx, order='C')
    labels = gmm.predict_proba(data_f).reshape(ne, nt, N_WT)
    return xr.DataArray(
        labels,
        coords={'ensemble': z500_anomaly.ensemble.values, 'time': z500_anomaly.time.values, 'wt': np.arange(1, N_WT + 1)},
        dims=('ensemble', 'time', 'wt'),
        name='wt_probability',
    )

def build_wrgmm_precip(wt_probability):
    members = []
    for ens in wt_probability.ensemble.values:
        ens_prob = wt_probability.sel(ensemble=ens)
        daily_fields = []
        for current_time in ens_prob.time.values:
            ts = pd.Timestamp(current_time)
            profile_stack = xr.concat(
                [load_precip_profile(ts.month, wt).expand_dims(wt=[wt]) for wt in range(1, N_WT + 1)],
                dim='wt'
            ).transpose('wt', 'latitude', 'longitude')
            field = xr.dot(ens_prob.sel(time=current_time), profile_stack, dims='wt')
            daily_fields.append(field.expand_dims(time=[current_time]))
        member = xr.concat(daily_fields, dim='time').expand_dims(ensemble=[int(ens)])
        members.append(member)
    out = xr.concat(members, dim='ensemble').transpose(
        'ensemble', 'time', 'latitude', 'longitude'
    )
    out.name = 'precip_wrgmm'
    return out

def aggregate_periods(daily_precip, init_date):
    fields = []
    for period in PERIODS:
        start, end = get_period(init_date, period)
        field = daily_precip.sel(time=slice(start, end)).mean('time')
        field = field.expand_dims(period=[period])
        field = field.assign_coords(target_start=('period', [np.datetime64(start)]))
        field = field.assign_coords(target_end=('period', [np.datetime64(end)]))
        fields.append(field)
    return xr.concat(fields, dim='period')

def build_blended_monthly_precip(raw_monthly, wrgmm_monthly, hist_r):
    blended = raw_monthly * (1 - hist_r) + wrgmm_monthly * hist_r
    blended.name = 'precip_blended_monthly'
    return blended

def compute_tercile_probabilities(period_precip, model):
    zthr = scipy.stats.norm.ppf([1 / 3, 2 / 3])
    categories = ['below', 'normal', 'above']
    lead_probs = []

    init_month = pd.Timestamp(init_date).month
    for period in PERIODS:
        member_probs = []
        for member in CFSV2_MEMBERS:
            forecast = fourth_root(period_precip.sel(period=period, ensemble=member))
            clim = load_model_coeff(member, period, 'clim').sel(month=init_month)
            sd = load_model_coeff(member, period, 'sd').sel(month=init_month).clip(min=1e-6)
            corr = load_model_coeff(member, period, 'r').sel(month=init_month).clip(-0.9999, 0.9999)
            zf = (forecast - clim) / sd
            spread = xr.apply_ufunc(
                lambda x: np.sqrt(np.maximum(1 - x**2, 1e-6)), corr
            )
            poe = 1 - scipy.stats.norm.cdf(
                zthr[np.newaxis, np.newaxis, :],
                loc=zf.values[..., np.newaxis],
                scale=spread.values[..., np.newaxis],
            )
            prob_arr = np.stack(
                [1 - poe[..., 0], poe[..., 0] - poe[..., 1], poe[..., 1]], axis=0
            )
            da = xr.DataArray(
                prob_arr,
                coords={
                    'category': categories,
                    'lat': forecast.latitude.values,
                    'lon': forecast.longitude.values,
                },
                dims=('category', 'lat', 'lon'),
            ).expand_dims(ensemble=[member])
            member_probs.append(da)

        start, end = get_period(init_date, period)
        da = xr.concat(member_probs, dim='ensemble').expand_dims(period=[period])
        da = da.assign_coords(target_start=('period', [np.datetime64(start)]))
        da = da.assign_coords(target_end=('period', [np.datetime64(end)]))
        lead_probs.append(da)

    member_prob = xr.concat(lead_probs, dim='period')
    mean_prob = member_prob.mean(dim='ensemble').rename('tercile_probability')
    return member_prob, mean_prob

def run_model(model, init_date, save=True):
    out_dir = OUTPUT_DIR / init_date / model
    out_dir.mkdir(parents=True, exist_ok=True)

    z500_daily = load_cfsv2_z500(init_date)
    z500_regrid = regrid_to_core_grid(z500_daily, core_clim)
    z500_anomaly = make_daily_anomaly(z500_regrid, core_clim)
    print('Done: z500 process')
    wt_probability = predict_probabilities(z500_anomaly, gmm)
    precip_wrgmm = build_wrgmm_precip(wt_probability)
    monthly_precip = aggregate_periods(precip_wrgmm, init_date)
    print('Done: WRGMM precip process')
    raw_tp_daily = load_cfsv2_tp(init_date)
    raw_tp_regrid = regrid_tp_to_obs_grid(raw_tp_daily, obs_grid)
    raw_tp_monthly = aggregate_periods(raw_tp_regrid, init_date)
    print('Done: raw precip process')
    blended_monthly = build_blended_monthly_precip(raw_tp_monthly, monthly_precip, hist_r)
    blended_member, blended_mean = compute_tercile_probabilities(blended_monthly, model)
    
    outputs = {
        'wt_probability': wt_probability,
        'wrgmm_precip': monthly_precip,
        'raw_precip': raw_tp_monthly,
        'exwrgmm_precip': blended_monthly,
        'tercile_member': blended_member,
        'tercile_mean': blended_mean,
    }

    if save:
        wt_probability.to_netcdf(out_dir / f'{model}_{init_date}_wt_probability.nc')
        monthly_precip.to_netcdf(out_dir / f'{model}_{init_date}_precip_wrgmm_monthly.nc')
        raw_tp_monthly.to_netcdf(out_dir / f'{model}_{init_date}_precip_raw_monthly.nc')
        blended_monthly.to_netcdf(out_dir / f'{model}_{init_date}_precip_exwrgmm_monthly.nc')
        blended_member.to_netcdf(out_dir / f'{model}_{init_date}_tercile_probability_member.nc')
        blended_mean.to_netcdf(out_dir / f'{model}_{init_date}_tercile_probability_mean.nc')

    return outputs

core_clim = load_core_climatology()
hist_r = load_hist_r_weight()
obs_grid = open_dataarray(OBS_CLIM_PATH, 'precip')
gmm = load_gmm()

results = {MODEL_NAME: run_model(MODEL_NAME, init_date, save=True)}


def build_outlook_masks(prob_da, equal_chance_threshold=0.40):
    prob_da = prob_da.transpose('category', 'lat', 'lon')
    valid_mask = prob_da.notnull().any('category')
    prob_filled = prob_da.fillna(-999.0)
    max_prob = xr.where(valid_mask, prob_da.max('category', skipna=True), np.nan)
    dom_idx = prob_filled.argmax('category')
    category_names = prob_da.category.values
    dominant = xr.DataArray(
        np.asarray(category_names)[dom_idx.values],
        coords={'lat': prob_da.lat, 'lon': prob_da.lon},
        dims=('lat', 'lon')
    )
    dominant = xr.where(valid_mask, dominant, 'ec')
    ec_mask = (~valid_mask) | (max_prob < equal_chance_threshold)
    masks = {
        'above': xr.where((dominant == 'above') & (~ec_mask), max_prob, np.nan),
        'normal': xr.where((dominant == 'normal') & (~ec_mask), max_prob, np.nan),
        'below': xr.where((dominant == 'below') & (~ec_mask), max_prob, np.nan),
        'ec': xr.where(ec_mask, 1.0, np.nan),
    }
    return masks, dominant, max_prob

def valid_label_from_plot_da(prob_da):
    start = pd.Timestamp(prob_da.target_start.values)
    end = pd.Timestamp(prob_da.target_end.values)
    return f'{start:%B %d, %Y} - {end:%B %d, %Y}'

def add_category_label(ax, mask, text):
    valid = mask.where(~np.isnan(mask), drop=True)
    if valid.size == 0:
        return
    peak = valid.stack(points=('lat', 'lon')).idxmax('points').item()
    lat, lon = peak
    ax.text(
        float(lon), float(lat), text,
        transform=ccrs.PlateCarree(),
        ha='center', va='center', fontsize=16, fontweight='bold', color='black',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')]
    )

def plot_tercile_outlook(prob_da, model, lead_month, issued_label, equal_chance_threshold=0.40):
    masks, dominant, max_prob = build_outlook_masks(prob_da, equal_chance_threshold=equal_chance_threshold)

    above_levels = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    below_levels = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    normal_levels = [0.40, 0.50, 0.60, 0.70, 1.01]

    above_cmap = ListedColormap(['#b7dca8', '#93cf74', '#63bf46', '#10a52a', '#0a861f', '#2f6b08'])
    below_cmap = ListedColormap(['#f7dda0', '#e8ba55', '#d9923b', '#a75d33', '#8b5a2b', '#5b3a36'])
    normal_cmap = ListedColormap(['#dddddd', '#bdbdbd', '#8f8f8f', '#666666'])

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-126, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)

    country_path = shapereader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
    country_reader = shapereader.Reader(country_path)
    conus_box = box(-126, 24, -66, 50)
    usa_geoms = []
    for rec in country_reader.records():
        if rec.attributes.get('ADM0_A3') == 'USA':
            geom = rec.geometry.intersection(conus_box)
            if not geom.is_empty:
                usa_geoms.append(geom)

    ax.add_geometries(usa_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='0.35', linewidth=1.0)
    reader = shapereader.Reader(BASE_DIR / 'cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    state_geoms = []
    for geom in reader.geometries():
        clipped = geom.intersection(conus_box)
        if not clipped.is_empty:
            state_geoms.append(clipped)
    ax.add_geometries(state_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='0.35', linewidth=0.8)

    lon2d, lat2d = np.meshgrid(prob_da.lon.values, prob_da.lat.values)
    ax.contourf(lon2d, lat2d, masks['ec'], levels=[0.5, 1.5], colors=['#ffffff'], transform=ccrs.PlateCarree())
    ax.contourf(lon2d, lat2d, masks['normal'], levels=normal_levels, cmap=normal_cmap, extend='max', transform=ccrs.PlateCarree())
    ax.contourf(lon2d, lat2d, masks['above'], levels=above_levels, cmap=above_cmap, extend='max', transform=ccrs.PlateCarree())
    ax.contourf(lon2d, lat2d, masks['below'], levels=below_levels, cmap=below_cmap, extend='max', transform=ccrs.PlateCarree())

    add_category_label(ax, masks['above'], 'Above')
    add_category_label(ax, masks['below'], 'Below')
    add_category_label(ax, masks['normal'], 'Near\nNormal')
    if np.isfinite(masks['ec'].values).any():
        ax.text(-93, 46, 'Equal\nChances', transform=ccrs.PlateCarree(), ha='center', va='center',
                fontsize=16, fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # ax.set_title(f'{model} Precipitation Outlook', fontsize=34, pad=20)
    valid_txt = valid_label_from_plot_da(prob_da)
    fig.text(0.50, 0.92, f'{model} Precipitation Outlook', ha='center', fontsize=34, fontweight='bold')
    fig.text(0.50, 0.87, f'Valid: {valid_txt}', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.50, 0.835, f'Issued: {issued_label}', ha='center', fontsize=18)
    fig.text(0.50, 0.15, 'Probability (Percent Chance)', ha='center', fontsize=14, fontweight='bold')

    above_handles = [
        mpatches.Patch(color='#b7dca8', label='Above 40-50%'),
        mpatches.Patch(color='#93cf74', label='Above 50-60%'),
        mpatches.Patch(color='#63bf46', label='Above 60-70%'),
        mpatches.Patch(color='#10a52a', label='Above 70-80%'),
        mpatches.Patch(color='#0a861f', label='Above 80-90%'),
        mpatches.Patch(color='#2f6b08', label='Above 90-100%'),
    ]

    center_handles = [
        mpatches.Patch(color='#bdbdbd', label='Near Normal 40-50%'),
        mpatches.Patch(facecolor='white', edgecolor='0.5', label='Equal Chances'),
    ]

    below_handles = [
        mpatches.Patch(color='#f7dda0', label='Below 40-50%'),
        mpatches.Patch(color='#e8ba55', label='Below 50-60%'),
        mpatches.Patch(color='#d9923b', label='Below 60-70%'),
        mpatches.Patch(color='#a75d33', label='Below 70-80%'),
        mpatches.Patch(color='#8b5a2b', label='Below 80-90%'),
        mpatches.Patch(color='#5b3a36', label='Below 90-100%'),
    ]

    fig.legend(handles=above_handles, loc='upper center', bbox_to_anchor=(0.36, 0.12), ncol=1, fontsize=11, frameon=False)
    fig.legend(handles=center_handles, loc='upper center', bbox_to_anchor=(0.50, 0.12), ncol=1, fontsize=11, frameon=False)
    fig.legend(handles=below_handles, loc='upper center', bbox_to_anchor=(0.64, 0.12), ncol=1, fontsize=11, frameon=False)

    return fig, ax, masks, dominant, max_prob

def monthly_wt_probability(wt_probability, init_date, n_leads=N_LEADS):
    target_months = get_target_month_starts(init_date, n_leads=n_leads)
    lead_probs = []
    for lead_idx, target_month in enumerate(target_months, start=1):
        start = pd.Timestamp(target_month)
        end = start + pd.offsets.MonthEnd(0)
        prob = (wt_probability.sel(time=slice(start, end)).mean(dim=('ensemble', 'time')))
        prob = prob.expand_dims(lead_month=[lead_idx])
        prob = prob.assign_coords(target_time=('lead_month', [np.datetime64(start)]))
        lead_probs.append(prob)
    out = xr.concat(lead_probs, dim='lead_month')
    out.name = 'monthly_wt_probability'
    return out


def plot_all_model_wt_probability(results, init_date, model_names=(MODEL_NAME,), n_leads=N_LEADS):
    fig, axes = plt.subplots(1, len(model_names),figsize=(5.0 * len(model_names), 6),sharey=True)

    if len(model_names) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_names):
        wt_monthly = monthly_wt_probability(results[model]['wt_probability'],init_date,n_leads=n_leads)
        values = wt_monthly.transpose('lead_month', 'wt').values * 100.0
        leads = wt_monthly.lead_month.values
        bottom = np.zeros(len(leads))

        for wt in range(1, N_WT + 1):
            vals = values[:, wt - 1]
            ax.bar(leads,vals,bottom=bottom,color=WR_COLORS[wt],edgecolor='black',linewidth=1.0,label=WR_LABELS[wt])
            for x, y, b in zip(leads, vals, bottom):
                if y >= 3:
                    ax.text(x,b + y / 2,f'{y:.0f}',ha='center',va='center',color='white',fontsize=13,fontweight='bold')

            bottom += vals

        ax.set_title(model, fontsize=18, fontweight='bold')
        valid_times = pd.to_datetime(wt_monthly.target_time.values)
        xticklabels = [f"{t.year}-{t.month:02d}"for t in valid_times]
        ax.set_xticks(leads)
        ax.set_xticklabels(xticklabels, fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Weather Regime Probability (%)', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5, 0.93),ncol=5,fontsize=13,frameon=False)
    issued_label = pd.Timestamp(init_date).strftime('%B %d, %Y')
    fig.suptitle(f'Monthly Weather Regime Probability\nIssued: {issued_label}',fontsize=20,fontweight='bold',y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_png = batch_png_out_dir / f'All_models_lead1-4_wt_probability_{init_date}.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {out_png.name}')

equal_chance_threshold = 0.40
batch_png_out_dir = BASE_DIR / 'PNG' / init_date
batch_png_out_dir.mkdir(parents=True, exist_ok=True)

for batch_model in (MODEL_NAME,):
    if batch_model not in results:
        print(f'Skip: {batch_model} not in results')
        continue

    available_periods = [int(x) for x in results[batch_model]['tercile_mean'].period.values]
    for batch_period in available_periods:
        batch_da = results[batch_model]['tercile_mean'].sel(period=batch_period)
        batch_issued_label = pd.Timestamp(init_date).strftime('%B %d, %Y')

        fig, ax, masks, dominant, max_prob = plot_tercile_outlook(
            batch_da,
            model=batch_model,
            lead_month=batch_period,
            issued_label=batch_issued_label,
            equal_chance_threshold=equal_chance_threshold,
        )

        valid_label = valid_label_from_plot_da(batch_da).replace(' ', '_').replace(',', '').replace('-', 'to')
        out_png = batch_png_out_dir / f'{batch_model}_P{batch_period}_{valid_label}_tercile_outlook.png'

        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        # fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'Saved: {out_png.name}')

# P1/P5 precipitation outlook only. Monthly WR bar plot is not used here.
