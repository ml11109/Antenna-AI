"""
Data cleaning and management for antenna simulation datasets.
"""

from pathlib import Path

import pandas as pd

from data_constants import *

def clean_data():
    # Load files
    raw_data_dir, cleaned_data_path, default_params_path = [
        Path(p) for p in [RAW_DATA_DIRECTORY, CLEANED_DATA_PATH, DEFAULT_PARAMS_PATH]]

    df_list = []

    for path in raw_data_dir.iterdir():
        df_list.append(pd.read_csv(path))

    default_df = pd.read_csv(default_params_path)

    # Remove units and rename
    def refactor_df(df):
        new_df = pd.DataFrame()

        for col, ser in df.copy().items():
            if ser.dtype == 'O':
                # Ignore values without numbers
                if ser.iloc[0].isalpha():
                    continue

                # Extract starting numerical values
                ser = ser.str.extract(r'([-+]?\d*\.?\d+)')[0].astype(float)

            if '[' in col and ']' in col:
                unit = col[col.index('[') + 1: col.index(']')]
                col = col[:col.index('[') - 1]

                # Convert units
                if unit:
                    ser *= CONVERSIONS[unit]

            new_df[col] = ser

        new_df.rename(columns=RENAME, inplace=True)

        return new_df

    for i, df in enumerate(df_list):
        df_list[i] = refactor_df(df)

    default_df = refactor_df(default_df)

    # Fill in default params
    full_data = pd.concat(df_list, axis=0)

    for param in set(default_df.columns) & set(INPUTS):
        if param in full_data.columns:
            full_data.loc[full_data[param].isna(), param] = default_df.loc[0, param]
        else:
            full_data[param] = default_df.loc[0, param]

    # Refactor
    selected_data = full_data[INPUTS + [OUTPUT]].dropna()

    # Save data
    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    selected_data.to_csv(cleaned_data_path, index=False)

    return selected_data

clean_data()
