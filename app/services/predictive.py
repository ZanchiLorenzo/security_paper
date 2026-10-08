import pandas as pd
import numpy as np
import calendar
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib
import logging

logger = logging.getLogger(__name__)

def predict_pax(df: pd.DataFrame, path: str) -> pd.DataFrame:

    df = df.copy()

    if df.empty:
        df["pax_standard"] = pd.Series(dtype="int64")
        return df

    required_source_columns = {
        "fdatetime",
        "car",
        "seats",
        "apt",
        "ac",
    }

    missing_columns = (required_source_columns - set(df.columns))

    if missing_columns:
        raise KeyError(f"Missing prediction columns: {sorted(missing_columns)}")

    df["fdatetime"] = pd.to_datetime(df["fdatetime"], errors="coerce")

    if df["fdatetime"].isna().any():
        raise ValueError("Some values in 'fdatetime' are invalid.")

    invalid_seats = (df["seats"].isna() | (df["seats"] <= 0))

    if invalid_seats.any():
        invalid_flights = df.loc[invalid_seats, "nflt"].tolist()
        raise ValueError(f"Missing or invalid aircraft capacity for flights: {invalid_flights}")

    # Load trained model + preprocessing objects
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Unable to load model bundle from {path}") from e

    required_keys = {
        "model",
        "apt_encoder",
        "other_encoder",
        "features",
        "apt_column",
        "other_categorical_columns",
        "numeric_columns",
    }

    missing_keys = required_keys - bundle.keys()

    if missing_keys:
        raise KeyError(f"Invalid model bundle. Missing keys: {sorted(missing_keys)}")

    model = bundle["model"]
    apt_encoder = bundle["apt_encoder"]
    other_encoder = bundle["other_encoder"]
    features = bundle["features"]
    apt_column = bundle["apt_column"]
    other_categorical_columns = bundle["other_categorical_columns"]
    numeric_columns = bundle["numeric_columns"]

    # Temporal features
    df["month"] = df["fdatetime"].dt.month_name(locale="en_US")
    df["week_day"] = df["fdatetime"].dt.day_name(locale="en_US")

    hour_decimal = (df["fdatetime"].dt.hour + df["fdatetime"].dt.minute / 60)

    df["sin_hour"] = np.sin(
        2 * np.pi * hour_decimal / 24
    )

    df["cos_hour"] = np.cos(
        2 * np.pi * hour_decimal / 24
    )

    # Prepare model input
    X_pred = df[features].copy()

    # Airport encoding
    X_pred_apt = apt_encoder.transform(X_pred[apt_column])
    apt_names = apt_encoder.get_feature_names_out(apt_column)

    # Other categorical encoding
    X_pred_other = other_encoder.transform(X_pred[other_categorical_columns])
    other_names = other_encoder.get_feature_names_out(other_categorical_columns)

    # Assemble final matrix
    X_pred_model = pd.concat(
        [
            X_pred[numeric_columns].reset_index(drop=True),

            pd.DataFrame(
                X_pred_apt,
                columns=apt_names,
            ),

            pd.DataFrame(
                X_pred_other,
                columns=other_names,
            ),
        ],
        axis=1,
    )

    # Same column order used during training
    X_pred_model = X_pred_model[model.feature_names_in_]

    # Predict pax
    pax_pred = model.predict(X_pred_model)
    pax_pred = np.rint(pax_pred).astype(int)

    df["pax_standard"] = pax_pred
    df["flight_type"] = "standard"

    cols_to_drop = [
        "country",
        "schh",
        "schm",
        "month",
        "week_day",
        "sin_hour",
        "cos_hour",
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df




# roba vecchia--------

def standardize_df(df):
    temp= df.copy()

    if 'fdatetime' not in temp.columns:
        if 'schtime' in temp.columns and 'date' in temp.columns:
            temp["fdatetime"] = pd.to_datetime(temp["date"].astype(str) + " " + temp["schtime"], errors="coerce")

    temp['date']= pd.to_datetime(temp['date'])
    temp['month']= temp['date'].dt.month
    temp['week_day']= temp['date'].dt.day_name()
    temp['month']= temp['month'].apply(lambda x: calendar.month_name[x])

    regions = {
        'Northern Europe': {'ARN', 'GOT', 'TRF', 'OSL', 'BGO', 'LPP', 'HEL', 'CPH', 'BLL', 'TLL', 'RIX', 'VNO', 'KUN'},
        'Western Europe': {'CDG', 'BVA', 'LDE', 'MRS', 'BOD', 'ORY', 'TLS', 'TUF', 'NCE', 'TLN', 'LYS', 'AJA', 'NTE', 
                           'FRA', 'BER', 'CGN', 'NRN', 'HAM', 'DUS', 'HHN', 'LEJ', 'FMM', 'HAJ', 'NUE', 'FMO', 'MUC', 'STR', 'FKB', 'MHG', 'PAD', 
                           'EIN', 'AMS', 'RTM', 'CRL', 'LUX', 'VIE', 
                           'EMA', 'MAN', 'STN', 'BRS', 'BHX', 'EDI', 'LGW', 'LPL', 'LTN', 'NCL', 'MME', 'HUY', 'NWI', 'BOH', 'NQY', 'LHR', 'BFS', 'SOU','BQH', 
                           'NOC', 'DUB', 'ORK', 'SNN'},
        'Southern Europe': {'PSR', 'NAP', 'TPS', 'BRI', 'AHO', 'SUF', 'CIY', 'CTA', 'CAG', 'BDS', 'PMO', 'OLB', 'CRV', 'TRS', 'FCO', 'FLR', 'PEG', 'PSA', 'LIN', 'MXP', 'CIA', 'LMP', 'PNL', 'VCE', 'BLQ', 'AOI', 'VRN', 'BGY', 'RMI', 
                            'VIT', 'ACE', 'VLC', 'AGP', 'MAD', 'BCN', 'LPA', 'FUE', 'TFS', 'ALC', 'SCQ', 'ZAZ', 'SVQ', 'PMI', 'SDR', 'IBZ', 'MAH', 'OVD','GRX', 
                            'LIS', 'OPO', 'PXO', 'FAO', 'FNC', 'BYJ', 'ATH', 'SKG', 'JTR', 'RHO', 'CFU', 'HER', 'CHQ', 'PVK', 'EFL','KLX', 'KGS', 'ZTH', 'AOK', 'JMK',
                            'ZAG', 'ZAD', 'RJK', 'MLA', 'TIA', 'PFO', 'LCA'},
        'Eastern Europe': {'OTP', 'TSR', 'SCV', 'OMR', 'CRA', 'BCM', 'CLJ', 'IAS', 'BAY', 'SOF', 'VAR', 'POZ', 'KRK', 'GDN', 'KTW', 'WMI', 'WAW', 'WRO', 'LCJ', 'RZE','LUZ',
                           'BTS', 'PRG', 'BRQ', 'BUD', 'INI', 'BNX', 'OMO'},
        'Africa': {'CAI', 'SSH', 'HBE', 'RMF', 'CMN', 'RAK', 'TNG', 'FEZ', 'AGA', 'DSS', 'SID', 'DJE', 'MUH'},
        'Middle East': {'ATQ', 'TLV', 'SAW', 'KIV', 'VKO', 'AYT', 'SHJ', 'ETM', 'TBS', 'DXB', 'IST'},
    }

    def group_countries_by_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return 'Other'

    carrier_groups = {
        'low_cost_east_eu': {'0B', '2B', 'A9', 'H4', 'PC', 'ZB', 'CY', 'RO', 'SM', 'VF', 'DN', 'H7', 'XC'},
        'arab': {'3O', 'E5', 'FZ', 'G9', 'UJ', 'IZ', 'FT', 'LY', 'AL', 'NP', 'NE'},
        'generic_eu': {'AP', 'NO', 'EW', 'V7', 'VY', 'XZ', 'YW', 'KM', 'A3', 'EN', 'AZ'},
        'north_eu': {'DY', 'HV', 'TB', 'LH', 'X9', 'MYX', 'KLJ', '6I', 'D8'},
        'ryanair': {'FR', 'RK', 'RR'},
        'easyjet': {'EC', 'U2'},
        'wizz_air': {'W4', 'W6'},
        'turkish': {'TK', 'TI'},
        'charter': {'6Y', 'BUC', 'HT', 'PWF', 'YU', 'C3', 'T3', 'DK', 'BZ', 'BQ', 'L9', 'MV'},
    }

    def group_carriers(carrier):
        for group, carriers in carrier_groups.items():
            if carrier in carriers:
                return group
        return 'other'

    h = temp["fdatetime"].dt.hour
    conditions = [
        (h >= 5) & (h < 8),
        (h >= 8) & (h < 12),
        (h >= 12) & (h < 17),
        (h >= 17) & (h < 24),
        (h >= 0) & (h < 5),
    ]

    choices = [
        "early morning",
        "morning",
        "afternoon",
        "evening",
        "past midnight",
    ]
    

    temp["period_of_day"] = np.select(conditions, choices, default="unknown")

    temp['region'] = temp['apt'].apply(group_countries_by_region)
    
    temp['carrier_group'] = temp['car'].apply(group_carriers)

    temp2= temp[['month', 'week_day', 'period_of_day', 'carrier_group', 'car', 'region', 'seats'] + 
           [col for col in temp.columns if col not in ['month', 'week_day', 'period_of_day', 'carrier_group', 'car', 'region', 'seats']]].copy()

    return temp2

def train_pax_prediction(df):
    # remove outliers
    df1 = df[
        (df['pax'] != 0) &
        (df['luggage'] != 0) &
        (df['luggage'] <= np.percentile(df['luggage'], 99)) &
        (df['luggage'] >= np.percentile(df['luggage'], 1)) &
        (df['pax'] <= df['seats'])
    ]

    # exclude car with very few samples
    carrier_counts = df1['car'].value_counts()
    valid_carriers = carrier_counts[carrier_counts >= 10].index
    df1 = df1[df1['car'].isin(valid_carriers)]

    df1= df1[['month', 'week_day', 'period_of_day', 'carrier_group', 'region', 'seats', 'pax']]

    x= df1.drop(columns=["pax"])
    y= df1["pax"]

    x_encoded= pd.get_dummies(x, columns=['month', 'week_day', 'period_of_day', 'carrier_group', 'region'], drop_first=True)
    x_encoded= x_encoded.astype(float)

    x_train, x_test, y_train, y_test= train_test_split(x_encoded, y, test_size=0.3, random_state=10)
    xgb = XGBRegressor(random_state=10)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)
    best_xgb_model= grid.best_estimator_

    # print kpis
    # train
    y_pred_xgb_train= best_xgb_model.predict(x_train)
    mae_xgb_train = mean_absolute_error(y_train, y_pred_xgb_train)
    r2_xgb_train = r2_score(y_train, y_pred_xgb_train)

    # test
    y_pred_xgb_test= best_xgb_model.predict(x_test)
    mae_xgb_test = mean_absolute_error(y_test, y_pred_xgb_test)
    r2_xgb_test = r2_score(y_test, y_pred_xgb_test)

    logger.info(f"Pax model Train MAE: {mae_xgb_train:.2f}, R2: {r2_xgb_train:.4f}")
    logger.info(f"Pax model Test MAE: {mae_xgb_test:.2f}, R2: {r2_xgb_test:.4f}")

    return best_xgb_model

def train_luggage_prediction(df):
    # remove outliers
    df1 = df[
        (df['pax'] != 0) &
        (df['luggage'] != 0) &
        (df['luggage'] <= np.percentile(df['luggage'], 99)) &
        (df['luggage'] >= np.percentile(df['luggage'], 1)) &
        (df['pax'] <= df['seats'])
    ]

    # exclude car with very few samples
    carrier_counts = df1['car'].value_counts()
    valid_carriers = carrier_counts[carrier_counts >= 10].index
    df1 = df1[df1['car'].isin(valid_carriers)]

    df1["luggage_ratio"] = df1["luggage"] / df1["pax"]

    df1= df1[['month', 'week_day', 'period_of_day', 'carrier_group', 'region', 'seats', 'pax', 'luggage_ratio']]

    x= df1.drop(columns=["luggage_ratio"])
    y= df1["luggage_ratio"]

    x_encoded= pd.get_dummies(x, columns=['month', 'week_day', 'period_of_day', 'carrier_group', 'region'], drop_first=True)
    x_encoded= x_encoded.astype(float)

    x_train, x_test, y_train, y_test= train_test_split(x_encoded, y, test_size=0.3, random_state=10)
    xgb = XGBRegressor(random_state=10)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)
    best_xgb_model= grid.best_estimator_

    # print kpis
    # train
    y_pred_xgb_train= best_xgb_model.predict(x_train)
    mae_xgb_train = mean_absolute_error(y_train, y_pred_xgb_train)
    r2_xgb_train = r2_score(y_train, y_pred_xgb_train)

    # test
    y_pred_xgb_test= best_xgb_model.predict(x_test)
    mae_xgb_test = mean_absolute_error(y_test, y_pred_xgb_test)
    r2_xgb_test = r2_score(y_test, y_pred_xgb_test)

    logger.info(f"Luggage model Train MAE: {mae_xgb_train:.2f}, R2: {r2_xgb_train:.4f}")
    logger.info(f"Luggage model Test MAE: {mae_xgb_test:.2f}, R2: {r2_xgb_test:.4f}")

    return best_xgb_model

def predict_pax0(model, df):
    x = df.drop(columns=["pax"], errors='ignore')
    x_encoded = pd.get_dummies(x, columns=['month', 'week_day', 'period_of_day', 'carrier_group', 'region'], drop_first=True)
    for col in x_encoded.columns:
        if x_encoded[col].dtype == 'uint8' or x_encoded[col].dtype == 'bool':
            x_encoded[col] = x_encoded[col].astype(int)

    expected_cols = model.get_booster().feature_names

    for col in expected_cols:
        if col not in x_encoded.columns:
            x_encoded[col] = 0

    extra_cols = set(x_encoded.columns) - set(expected_cols)
    if extra_cols:
        x_encoded = x_encoded.drop(columns=extra_cols)

    x_encoded = x_encoded[expected_cols]
    predicted_pax = model.predict(x_encoded)
    predicted_pax = np.clip(predicted_pax, 0, None)
    predicted_pax = np.ceil(predicted_pax).astype(int)  

    return predicted_pax

def predict_luggage(model, df):
    x = df.drop(columns=["luggage"], errors='ignore')
    x_encoded = pd.get_dummies(x, columns=['month', 'week_day', 'period_of_day', 'carrier_group', 'region'], drop_first=True)
    for col in x_encoded.columns:
        if x_encoded[col].dtype == 'uint8' or x_encoded[col].dtype == 'bool':
            x_encoded[col] = x_encoded[col].astype(int)

    expected_cols = model.get_booster().feature_names

    for col in expected_cols:
        if col not in x_encoded.columns:
            x_encoded[col] = 0

    extra_cols = set(x_encoded.columns) - set(expected_cols)
    if extra_cols:
        x_encoded = x_encoded.drop(columns=extra_cols)

    x_encoded = x_encoded[expected_cols]
    predicted_luggage = model.predict(x_encoded)
    predicted_luggage = np.clip(predicted_luggage, 0, 1)

    return predicted_luggage

def apply_predictions_to_schedule(payload, schedule):

    df = schedule.copy()
    
    pax_pred_model = payload.pax_pred_model
    luggage_pred_model = payload.luggage_pred_model

    if 'pax' not in df.columns:
        df['pax'] = np.nan
    if 'pax_checked' not in df.columns:
        df['pax_checked'] = np.nan

    # identify rows that need predictions
    need_pax = df['pax'].isna()
    need_checked = df['pax_checked'].isna()
    need_any = need_pax | need_checked

    # no need of predictions, return df with carry-on computed
    if not need_any.any():
        df['pax_carryon'] = df['pax'] - df['pax_checked']
        return df

    # apply predictions to rows that need them, if any
    df_std = standardize_df(df)
    if need_pax.any():
        df.loc[need_pax, 'pax'] = predict_pax(pax_pred_model, df_std.loc[need_pax])

    if need_checked.any():
        df.loc[need_checked, 'luggage_ratio'] = predict_luggage(luggage_pred_model, df_std.loc[need_checked])
        df.loc[need_checked, 'luggage_ratio'] = df.loc[need_checked, 'luggage_ratio'].clip(lower=0, upper=1)
        df.loc[need_checked, 'pax_checked'] = df.loc[need_checked, 'pax'] * df.loc[need_checked, 'luggage_ratio']

    # ensure pax_checked is not greater than pax and round to integers
    mask = df['pax'].notna() & df['pax_checked'].notna()
    df.loc[mask, 'pax_checked'] = df.loc[mask, 'pax_checked'].clip(upper=df.loc[mask, 'pax'])
    df[['pax', 'pax_checked']] = df[['pax', 'pax_checked']].round().astype('Int64')

    df['pax_carryon'] = df['pax'] - df['pax_checked']

    return df