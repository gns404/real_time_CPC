from pathlib import Path
import pickle
import os
import csv
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
CFS_DIR = BASE_DIR / "CFSv2"
PROFILE_DIR = BASE_DIR / "precip_profile"
GMM_PATH = BASE_DIR / "gmm_CORe.sav"
OUTPUT_DIR = CFS_DIR / "WR_WRGMM"

MEMBERS = tuple(
    (day, hour) for day in (4, 5, 6) for hour in (0, 6, 12, 18)
)

def makeWT(yr, mn):
    gmm = pickle.load(open(GMM_PATH, 'rb'))

    for ensemble, (day, hour) in enumerate(MEMBERS):
        init = f"{yr}{mn}{day:02d}{hour:02d}"
  
        filepath = CFS_DIR / f"z500_{day:02d}{hour:02d}" / f"z500.{init}.anomaly.nc"
        save_dir = CFS_DIR / f"z500_{day:02d}{hour:02d}" / f"weatherTypes_CFSR/"
        os.makedirs(save_dir, exist_ok=True)

        outputFile = f"WT.{yr}{mn}.csv"
        savepath = os.path.join(save_dir, outputFile)
        try:
            z500_da = xr.open_dataset(filepath).z500

            data = z500_da.values
            nt,ny,nx = data.shape
            data = np.reshape(data, [nt, ny*nx], order='F')

            labels = gmm.predict_proba(data)
            # output results
            with open(savepath, 'w', newline = '') as csvfile:
                my_writer = csv.writer(csvfile, delimiter = ',')
                for i in range(len(labels)):
                    row_data = [-999.9]*(5+1)
                    row_data[0] = z500_da.time.values[i] - z500_da.time.values[0]
                    for j in range(5):
                        row_data[j+1] = labels[i][j]
                    my_writer.writerow(row_data)

        except FileNotFoundError:print(f"{filepath} file not found.")

    return 

for year in ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]:
    for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
        makeWT(year, month)


def WRGMM(arg):
    yr, mn, lon, lat = arg

    for ensemble, (day, hour) in enumerate(MEMBERS):
        init = f"{yr}{mn}{day:02d}{hour:02d}"

        WTpath = CFS_DIR / f"z500_{day:02d}{hour:02d}" / f"weatherTypes_CFSR" / f"WT.{yr}{mn}.csv"
        save_dir = CFS_DIR / f"z500_{day:02d}{hour:02d}" / f"precip_WRGMM_CFSR/"
        os.makedirs(save_dir, exist_ok=True)

        outputFile = f"precip.WRGMM.{yr}{mn}.nc"
        savepath = os.path.join(save_dir, outputFile)

        try:
            with open(WTpath) as file:
                csvreader = csv.reader(file)       
                forecastWTs = []
                for row in csvreader:
                    forecastWTs.append(row)    

            initaldate = datetime(int(yr), int(mn), day)
            new_time = pd.date_range(start=initaldate, periods=len(forecastWTs), freq='D')
            precip_WRGMM = []
            ld = 0
            for row in forecastWTs:
                temp = np.zeros((len(lat), len(lon)))
                month = new_time[ld].month
                if month < 10:
                    ld_mn = "0"+str(month)
                else:
                    ld_mn = str(month)
                for i in range(5):
                    profile_path = PROFILE_DIR / f"Precip{ld_mn}WT{i+1}.nc"
                    with xr.open_dataset(profile_path) as ds:
                        WT_precip = ds["precip"].values
                    temp += float(row[i+1]) * WT_precip
                    
                precip_WRGMM.append(temp)
                ld += 1

            if os.path.exists(savepath):os.remove(savepath)
            precip_WRGMM = xr.DataArray(precip_WRGMM, coords=[new_time, lat, lon], dims=["time", "lat", "lon"])
            precip_WRGMM.to_netcdf(savepath)
        except FileNotFoundError:
            print("file not found "+yr+mn)

filepath_precip= BASE_DIR / "cpc.198101-201012.cfsv2_native_grid.nc"
precip_da = xr.open_dataset(filepath_precip).precip
lon = precip_da.longitude
lat = precip_da.latitude

month = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for year in ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]:
    with mp.Pool(processes=12) as pool:
        arg_list = [(year, month[j], lon, lat) for j in range(len(month))]
        pool.map(WRGMM, arg_list)
