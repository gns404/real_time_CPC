import cdsapi
import os
from datetime import datetime, timedelta

client = cdsapi.Client()

now = datetime.now()
year = now.year
month = now.month

base_dir = "original_forecast"
date_str = f"{year}{month:02d}01"

# ---------------------------
# tp
# ---------------------------
dataset = "seasonal-original-single-levels"

save_dir = os.path.join(base_dir, date_str, "tp")
os.makedirs(save_dir, exist_ok=True)

leadtime_hours = [str(x) for x in range(24, 3121, 24)]

# ECMWF
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["total_precipitation"],
    "year": str(year),
    "month": f"{month:02d}",
    "day": ["01"],
    "leadtime_hour": leadtime_hours,
    "data_format": "netcdf"
}

filename = f"ECMWF_{date_str}.nc"
filepath = os.path.join(save_dir, filename)
client.retrieve(dataset, request).download(filepath)

print(f"{filepath} download completed!")

# ---------------------------
# z500
# ---------------------------
dataset = "seasonal-original-pressure-levels"

save_dir = os.path.join(base_dir, date_str, "z500")
os.makedirs(save_dir, exist_ok=True)

leadtime_hours = [str(x) for x in range(12, 3121, 12)]

# ECMWF
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["geopotential"],
    "pressure_level": ["500"],
    "year": str(year),
    "month": f"{month:02d}",
    "day": ["01"],
    "leadtime_hour": leadtime_hours,
    "data_format": "netcdf"
}

filename = f"ECMWF_{date_str}.nc"
filepath = os.path.join(save_dir, filename)
client.retrieve(dataset, request).download(filepath)

print(f"{filepath} download completed!")