"""Conservatively regrid CPC precipitation to a specific CFSv2 native grid.

The target is read directly from the supplied CFSv2 precipitation NetCDF.
Its native Gaussian latitude/longitude coordinates are retained exactly; no
regular 1-degree grid is constructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np
import xarray as xr

BASE_DIR = Path("/work05/home/jihun/real_time_CPC")
DEFAULT_INPUT = Path(
    "/work05/home/jihun/seasonal_forecast/reanalysis/cpc.198101-201012.nc"
)
DEFAULT_TEMPLATE = (
    BASE_DIR / "CFSv2/precip_0506/prate.2012030506.nc"
)
DEFAULT_OUTPUT = BASE_DIR / "cpc.198101-201012.cfsv2_native_grid.nc"


def coordinate_names(dataset: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in dataset.coords else "lat"
    lon_name = "longitude" if "longitude" in dataset.coords else "lon"
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError("Dataset has no recognizable latitude/longitude coordinates")
    return lat_name, lon_name


def validate_inputs(input_path: Path, template_path: Path, variable: str) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    if shutil.which("cdo") is None:
        raise RuntimeError("cdo is required but was not found")

    with xr.open_dataset(input_path) as source:
        if variable not in source.data_vars:
            raise KeyError(f"{variable!r} is not present in {input_path}")
        coordinate_names(source)
    with xr.open_dataset(template_path) as template:
        coordinate_names(template)


def regrid(
    input_path: Path,
    template_path: Path,
    output_path: Path,
    variable: str,
) -> None:
    """Run first-order conservative remapping using the NetCDF grid template."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.nc")
    temporary.unlink(missing_ok=True)
    command = [
        "cdo", "-O", "-L", "-f", "nc4", "-z", "zip_4",
        f"remapcon,{template_path}",
        f"-selname,{variable}",
        str(input_path),
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_grid(output_path: Path, template_path: Path, variable: str) -> None:
    with xr.open_dataset(template_path) as template, xr.open_dataset(output_path) as output:
        template_lat_name, template_lon_name = coordinate_names(template)
        output_lat_name, output_lon_name = coordinate_names(output)
        target_lat = template[template_lat_name].values
        target_lon = template[template_lon_name].values
        output_lat = output[output_lat_name].values
        output_lon = output[output_lon_name].values

        if not np.array_equal(output_lat, target_lat):
            max_error = float(np.max(np.abs(output_lat - target_lat)))
            raise ValueError(f"Output latitude does not match template; max error={max_error}")
        if not np.array_equal(output_lon, target_lon):
            max_error = float(np.max(np.abs(output_lon - target_lon)))
            raise ValueError(f"Output longitude does not match template; max error={max_error}")
        if variable not in output:
            raise KeyError(f"Output variable {variable!r} is missing")

        field = output[variable]
        expected = (target_lat.size, target_lon.size)
        actual = (field.sizes[output_lat_name], field.sizes[output_lon_name])
        if actual != expected:
            raise ValueError(f"Output grid shape {actual} != template {expected}")
        print(f"Wrote: {output_path}")
        print(
            f"Grid matches template exactly: {expected[0]} latitude x "
            f"{expected[1]} longitude"
        )
        print(
            f"latitude: {target_lat[0]:.12g} .. {target_lat[-1]:.12g}; "
            f"longitude: {target_lon[0]:.12g} .. {target_lon[-1]:.12g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variable", default="precip")
    args = parser.parse_args()

    validate_inputs(args.input, args.template, args.variable)
    regrid(args.input, args.template, args.output, args.variable)
    verify_grid(args.output, args.template, args.variable)


if __name__ == "__main__":
    main()
