import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import requests
import re
import json
from app.services.utils import parse_schedule_time, assign_profile
from app.services.entities import Flight
from app.services.arrival_flow import get_arrival_sampler
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

# schedule preprocessing

def preprocess_schedule_FLR(dflt: pd.DataFrame, profile_specs: dict = None) -> pd.DataFrame:

    # Clean column names: remove newlines, trim whitespace, and replace multiple spaces with a single space
    dflt.columns = (dflt.columns
        .str.replace('\n', ' ', regex=False)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True))
    
    # Select and rename columns
    required_columns_map = {
        'Data': 'date',
        'Orario': 'schtime',
        'Numero Volo': 'nflt_num',
        'Codice IATA': 'car',
        'Scalo': 'apt',
        'Modello aerom.': 'ac',
        'Posti offerti': 'seats',
        'Mov.': 'dir',
    }

    # check on required columns
    for col in required_columns_map.keys():
        if col not in dflt.columns:
            raise KeyError(f"Missing required column '{col}' in schedule data.")
    
    dflt = dflt[list(required_columns_map.keys())].copy().rename(columns=required_columns_map)

    # Ensure types: date -> datetime
    dflt['date'] = pd.to_datetime(dflt['date'], format='%d/%m/%Y', errors='coerce')
    if dflt['date'].isnull().any():
        raise ValueError("Some dates in 'date' column could not be parsed.")
    # Ensure types: seats -> int
    dflt['seats'] = pd.to_numeric(dflt['seats'], errors='coerce').fillna(0).astype(int)

    # Extract date part for reference
    dflt['fdate'] = dflt['date'].dt.date

    # Ensure 'schtime' is string and zero-padded to 4 characters, then parse hours and minutes robustly
    dflt["schtime"], dflt["schh"], dflt["schm"] = parse_schedule_time(dflt["schtime"])

    # Compose full datetime
    dflt["fdatetime"] = dflt["date"] + pd.to_timedelta(dflt["schh"], unit="h") + pd.to_timedelta(dflt["schm"], unit="m")
    
    if dflt['fdatetime'].isnull().any():
        raise ValueError("Some scheduled datetimes could not be parsed from date and schtime.")

    # departing flights only
    dflt['dir'] = dflt['dir'].map({"A": "in", "P": "out"})
    dflt = dflt[dflt['dir']=='out'].copy()

    # Fill missing carrier codes
    missing_car = dflt['car'].isna().sum()
    if missing_car > 0:
        logger.info(f"Missing 'car' values: {missing_car} -> replaced with '_'")
    dflt['car'] = (dflt['car'].fillna('_')
        .astype('string')
        .str.strip()
        .str.upper()
        .replace('', '_')
    )

    dflt["apt"] = (
        dflt["apt"]
        .fillna("_")
        .astype("string")
        .str.strip()
        .str.upper()
        .replace("", "_")
    )

    dflt["nflt"] = (
        dflt["car"].fillna("_").astype("string").str.strip()
        + " "
        + dflt["nflt_num"].fillna("").astype("string").str.strip()
            .str.replace(r"^0+(?=.)", "", regex=True)
    )
    dflt.drop(columns=["nflt_num"], inplace=True)

    # Fill missing aircraft IATA codes
    missing_ac = dflt["ac"].isna().sum()
    if missing_ac > 0:
        logger.info(f"Missing 'ac' values: {missing_ac} -> replaced with '_'")
    dflt["ac"] = (dflt["ac"].fillna("_")
        .astype("string")
        .str.strip()
        .str.upper()
        .replace('', '_')
    )

    if profile_specs is None:
        profile_specs = {}
    dflt["flight_type"] = dflt.apply(assign_profile, axis=1, profile_specs=profile_specs)

    return dflt

# registry preprocessing

def _validate_processing_params(server_label: str, params: dict) -> None:
    """
    Validate processing distribution parameters for a server.
    """
    dist_name = params.get("dist")

    if dist_name == "gamma":
        required = {"shape", "scale"}
    elif dist_name == "gaussian":
        required = {"mean", "sigma"}
    elif dist_name == "exponential":
        required = {"scale"}
    else:
        raise ValueError(f"Unsupported processing distribution for server {server_label}: {dist_name}")

    missing = required - set(params)

    if missing:
        raise ValueError(f"Missing processing parameters for server {server_label}: {sorted(missing)}")

    if float(params.get("shift", 0)) < 0:
        raise ValueError(f"Shift for server {server_label} cannot be negative.")

    for parameter in required:
        if float(params[parameter]) <= 0:
            raise ValueError(f"{parameter} for server {server_label} must be positive.")

def _processing_unit_to_minutes(processing_unit: int) -> float:
    """Validate and return a processing window expressed in minutes."""

    if isinstance(processing_unit, bool) or not isinstance(processing_unit, int):
        raise TypeError("processing_unit must be an integer expressed in minutes.")

    if processing_unit <= 0:
        raise ValueError("processing_unit must be positive.")

    return float(processing_unit)

def build_processing_params_by_server(
    registry: dict,
    default_dist: str = "exponential",
    default_shape: float = 1.6,
    default_shift: float = 0.07,
    ) -> dict[int, dict]:
    """
    Build processing distribution parameters for each server.

    If a server does not explicitly define a distribution (`dist`),
    processing parameters are derived from:

        processing_rate
        processing_unit (in minutes)
    """

    if not isinstance(registry, dict) or not registry:
        raise ValueError("Registry must be a non-empty dictionary.")

    processing_params_by_server = {}
    labels = set()

    for server_id, server in registry.items():

        if not isinstance(server, dict):
            raise TypeError(f"Invalid registry entry {server_id}: expected dict.")

        # validate label
        label = str(server.get("label", "")).strip()

        if not label:
            raise ValueError(f"Missing server label for registry entry {server_id}.")
        if label in labels:
            raise ValueError(f"Duplicate server label in registry: {label}")
        labels.add(label)

        # validate priority
        priority = server.get("priority", 10**9)

        try:
            priority = int(priority)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Priority for server {label} must be an integer.") from exc

        if priority < 0:
            raise ValueError(f"Priority for server {label} must be non-negative.")

        params = server.copy()
        params["priority"] = priority

        dist_name = params.get("dist")

        # If no explicit distribution is supplied, derive Exponential
        # parameters from the average processing rate.

        if dist_name is None:
            processing_rate = params.get("processing_rate")
            processing_unit = params.get("processing_unit")

            if processing_rate is None:
                raise ValueError(f"Server {label} must contain either 'dist' or 'processing_rate'.")

            if processing_unit is None:
                raise ValueError(f"Server {label} with 'processing_rate' must also contain 'processing_unit'.")

            try:
                processing_rate = float(processing_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"processing_rate for server {label} must be numeric.") from exc

            if processing_rate <= 0:
                raise ValueError(f"processing_rate for server {label} must be positive.")

            # Convert the rate window to minutes
            rate_window_minutes = _processing_unit_to_minutes(processing_unit)

            # Mean processing time per job, expressed in minutes
            mean_processing_time = (rate_window_minutes / processing_rate)

            if default_dist == "gamma":
                params.update(
                    {
                        "dist": "gamma",
                        "shape": float(default_shape),
                        "scale": (mean_processing_time - default_shift) / default_shape,
                        "shift": float(default_shift),
                    }
                )
            else:  # exponential
                params.update(
                    {
                        "dist": "exponential",
                        "scale": mean_processing_time - default_shift,
                        "shift": float(default_shift),
                    }
                )

            if params["scale"] <= 0:
                raise ValueError(f"Invalid Gamma parameters for server {label}: shift must be lower than the mean processing time.")

        _validate_processing_params(server_label=label, params=params)

        processing_params_by_server[server_id] = params

    return processing_params_by_server

# flight demand preprocessing

def _falls_back_to_unk(arrival_model: dict, car, month, time_band, apt) -> bool:
    lookup = arrival_model["code_lookup"]
    values = [car, month, time_band, apt]
    return any(val not in lookup[col] for col, val in zip(arrival_model["cat_cols"], values))

def build_flight_demand(schedule: pd.DataFrame, arrival_model: dict) -> list[Flight]:

    df = schedule.copy()
    df["fdatetime"] = pd.to_datetime(df["fdatetime"], errors="coerce")

    df["month"] = df["fdatetime"].dt.month_name(locale="en_US")
    df["time_band"] = pd.cut(
        df["fdatetime"].dt.hour,
        bins=[-1, 5, 9, 13, 17, 21, 23],
        labels=["00-05", "06-09", "10-13", "14-17", "18-21", "22-23"],
    ).astype(str)

    pax_cols = [c for c in df.columns if c.startswith("pax_")]

    out = []
    unk_count = 0

    for row in df.itertuples(index=False):

        if pd.isna(row.fdatetime):
            continue

        pax_segments = {
            col.removeprefix("pax_"): int(0 if pd.isna(getattr(row, col)) else getattr(row, col))
            for col in pax_cols
        }

        flight_number = str(row.nflt).strip()
        departure_dt = pd.Timestamp(row.fdatetime).to_pydatetime()

        normalized_flight_number = "".join(flight_number.split())
        flight_key = f"{normalized_flight_number}_{departure_dt:%Y%m%d%H%M}"

        car = str(getattr(row, "car", ""))
        apt = str(getattr(row, "apt", ""))
        month = str(row.month)
        time_band = str(row.time_band)

        sampler = get_arrival_sampler(arrival_model, car, month, time_band, apt)

        if _falls_back_to_unk(arrival_model, car, month, time_band, apt):
            unk_count += 1

        out.append(
            Flight(
                flight_key=flight_key,
                flight_number=flight_number,
                departure_dt=departure_dt,
                pax_segments=pax_segments,
                flight_type=str(getattr(row, "flight_type", "standard")),
                carrier=str(getattr(row, "car", "")),
                destination=str(getattr(row, "apt", "")),
                aircraft=str(getattr(row, "ac", "")),
                arrival_sampler=sampler
            )
        )

    if out and unk_count / len(out) > 0.10:
        logger.warning(f"More than 10% of flights ({unk_count}/{len(out)}) fall back to the unknown arrival sampler. Check the arrival model coverage.")

    return out

