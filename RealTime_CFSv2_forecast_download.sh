#!/usr/bin/env bash
# Download the 12 real-time CFSv2 forecasts for the current month.
# Output layout is the same as RealTime_ecmwf_forecast_down.py:
# original_forecast/YYYYMM01/tp and original_forecast/YYYYMM01/z500

set -euo pipefail

BASE_DIR="/work05/home/jihun/real_time_CPC"
FORECAST_DIR="${BASE_DIR}/original_forecast"
# BASE_URL="https://www.ncei.noaa.gov/oa/prod-cfs-operational-forecast/index.html#time-series"
BASE_URL="https://www.ncei.noaa.gov/oa/prod-cfs-operational-forecast"

YEAR="${1:-$(date +%Y)}"
MONTH="${2:-$(date +%m)}"
MONTH=$(printf "%02d" "$((10#$MONTH))")

for command_name in wget; do
    command -v "$command_name" >/dev/null || {
        echo "Required command not found: $command_name" >&2
        exit 1
    }
done

if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "YEAR must contain four digits: $YEAR" >&2
    exit 1
fi
if (( 10#$MONTH < 1 || 10#$MONTH > 12 )); then
    echo "MONTH must be between 1 and 12: $MONTH" >&2
    exit 1
fi
download_one() {
    local variable="$1"
    local year="$2"
    local month="$3"
    local day="$4"
    local hour="$5"
    local init="${year}${month}${day}${hour}"
    local date_dir="${year}${month}01"
    local output_name
    local output_dir
    local grib_variable filename url temporary wget_status missing_log

    if [[ "$variable" == "precip" ]]; then
        grib_variable="prate"
        output_name="tp"
    else
        grib_variable="z500"
        output_name="z500"
    fi

    # RealTime_ecmwf_forecast_down.py와 동일한 월별 저장 구조
    output_dir="${FORECAST_DIR}/${date_dir}/${output_name}"
    # Keep the original NCEI filename unchanged.
    filename="${grib_variable}.01.${init}.daily"
    url="${BASE_URL}/time-series/${year}/${year}${month}/${year}${month}${day}/${init}/${grib_variable}.01.${init}.daily.grb2"
    mkdir -p "$output_dir"

    if [[ -s "${output_dir}/${filename}.grb2" ]]; then
        echo "Skip existing: ${variable} ${init}"
        return
    fi

    echo "Download: ${variable} ${init}"
    temporary="${output_dir}/.${filename}.grb2.part"
    wget_status=0
    wget --continue --tries=5 --timeout=60 -O "$temporary" "$url" || wget_status=$?
    if (( wget_status != 0 )); then
        rm -f "$temporary"
        if (( wget_status == 8 )); then
            missing_log="${BASE_DIR}/logs/missing_realtime_${year}${month}.txt"
            mkdir -p "${BASE_DIR}/logs"
            printf '%s variable=%s init=%s url=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variable" "$init" "$url" \
                >> "$missing_log"
            echo "Missing on NCEI (wget exit 8); continue: ${variable} ${init}" >&2
            return 0
        fi
        echo "wget failed with exit ${wget_status}: $url" >&2
        return "$wget_status"
    fi
    if [[ "$(head -c 4 "$temporary")" != "GRIB" ]]; then
        echo "Downloaded file is not GRIB (possibly an HTML error page): $url" >&2
        rm -f "$temporary"
        return 1
    fi
    mv "$temporary" "${output_dir}/${filename}.grb2"
    echo "Complete: ${variable} ${init} -> ${output_dir}/${filename}.grb2"
}

DATE_DIR="${YEAR}${MONTH}01"
echo "CFSv2 real-time download: ${DATE_DIR}"
echo "Output: ${FORECAST_DIR}/${DATE_DIR}/{tp,z500}"

for day in 04 05 06; do
    for hour in 00 06 12 18; do
        download_one precip "$YEAR" "$MONTH" "$day" "$hour"
        download_one z500 "$YEAR" "$MONTH" "$day" "$hour"
    done
done

echo "CFSv2 real-time download completed: ${DATE_DIR}"
