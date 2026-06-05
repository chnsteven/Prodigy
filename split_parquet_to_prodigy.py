#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMNS = ["job_id", "component_id", "app_name", "anom_name", "anom_input", "binary_anom"]

# These values must match the filters in reproducibility_experiments.py:
#   selected_apps   = ['exa', 'lammps', 'sw4', 'sw4lite']
#   selected_labels = ['none', 'memleak']
# We pick one representative value from each list so the downstream code
# treats this dataset the same way it treats the original Prodigy data.
APP_NAME   = "exa"
ANOM_NONE  = "none"
ANOM_ANOM  = "memleak"

# A single job identifier. The dataset has no real job concept, so we use a
# fixed string. Every row gets a unique component_id (its original row index),
# which guarantees that every (job_id, component_id) pair is unique — a hard
# requirement for the MultiIndex lookups in reproducibility_experiments.py.
JOB_ID = "0"


def build_label_df(value_series: pd.Series) -> pd.DataFrame:
    """
    Build a label DataFrame with the required columns from the raw 'value' column.

    Mapping to Prodigy schema
    ─────────────────────────
    - job_id       : fixed JOB_ID (no real job concept in this dataset)
    - component_id : original integer row index → unique per row
    - app_name     : APP_NAME  (must be in reproducibility_experiments.selected_apps)
    - anom_name    : ANOM_NONE / ANOM_ANOM  (must be in selected_labels)
    - anom_input   : raw anomaly signal, NaN filled to 0
    - binary_anom  : 0 = healthy, 1 = anomalous
    """
    anom_input  = value_series.fillna(0).astype(int)
    binary_anom = (anom_input != 0).astype(int)

    return pd.DataFrame(
        {
            "job_id":       JOB_ID,
            "component_id": value_series.index.astype(str),   # unique per row
            "app_name":     APP_NAME,
            "anom_name":    binary_anom.map({0: ANOM_NONE, 1: ANOM_ANOM}),
            "anom_input":   anom_input.values,
            "binary_anom":  binary_anom.values,
        },
        index=value_series.index,
    )


def split_and_save_dataset(
    parquet_path: Path,
    output_dir: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    parquet_path = Path(parquet_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    data = pd.read_parquet(parquet_path)
    if data.empty:
        raise ValueError(f"Loaded data is empty: {parquet_path}")

    # Last column is the raw label ('value')
    label_column = data.columns[-1]

    label_data   = build_label_df(data[label_column])
    feature_data = data.drop(columns=[label_column])

    # Embed job_id / component_id into the feature DataFrame so that
    # DataPipeline._read_data() returns a frame that reproducibility_experiments.py
    # can call set_index(['job_id', 'component_id']) on.
    feature_data = feature_data.copy()
    feature_data.insert(0, "component_id", label_data["component_id"].values)
    feature_data.insert(0, "job_id",       label_data["job_id"].values)

    # ── train / val / test split ──────────────────────────────────────────────
    x_temp, x_test, y_temp, y_test = train_test_split(
        feature_data,
        label_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    val_ratio_in_temp = val_size / (1 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_ratio_in_temp,
        random_state=random_state,
        shuffle=True,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    train_data_path  = output_dir / "prod_train_data.hdf"
    val_data_path    = output_dir / "prod_val_data.hdf"
    test_data_path   = output_dir / "prod_test_data.hdf"
    train_label_path = output_dir / "prod_train_label.csv"
    val_label_path   = output_dir / "prod_val_label.csv"
    test_label_path  = output_dir / "prod_test_label.csv"

    # HDF: store with default integer index; job_id/component_id are columns
    x_train.to_hdf(train_data_path, key="data", mode="w")
    x_val.to_hdf(val_data_path,     key="data", mode="w")
    x_test.to_hdf(test_data_path,   key="data", mode="w")

    # Labels: drop the DataFrame index (it's just the original row number)
    y_train.to_csv(train_label_path, index=False)
    y_val.to_csv(val_label_path,     index=False)
    y_test.to_csv(test_label_path,   index=False)

    print(f"Train : {len(x_train):>6}  rows")
    print(f"Val   : {len(x_val):>6}  rows")
    print(f"Test  : {len(x_test):>6}  rows")
    print()
    print(f"Saved train data  -> {train_data_path}")
    print(f"Saved val data    -> {val_data_path}")
    print(f"Saved test data   -> {test_data_path}")
    print(f"Saved train label -> {train_label_path}")
    print(f"Saved val label   -> {val_label_path}")
    print(f"Saved test label  -> {test_label_path}")


if __name__ == "__main__":
    root_dir     = Path(__file__).resolve().parent
    parquet_path = Path("../Dataset/ex100/0/2.parquet")

    split_and_save_dataset(
        parquet_path,
        root_dir,
        test_size=0.15,
        val_size=0.15,
    )
