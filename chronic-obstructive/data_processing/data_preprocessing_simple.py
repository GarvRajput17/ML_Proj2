"""
Simplified preprocessing for the chronic obstructive dataset.

- Drops patient_id (saves to ids_simple.csv)
- Encodes categorical columns (sex, oral_health_status, tartar_presence)
- Saves train/test processed CSVs for use in lightweight models or notebooks
"""

import os
import pandas as pd


CATEGORICAL_MAPPINGS = {
    "sex": {"M": 1, "F": 0},
    "oral_health_status": {"Y": 1, "N": 0},
    "tartar_presence": {"Y": 1, "N": 0},
}


def encode_categorical(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Encode categorical columns using the mappings above."""
    df = df.copy()

    if "sex" in df.columns:
        df["sex_encoded"] = df["sex"].map(CATEGORICAL_MAPPINGS["sex"])
        df = df.drop(columns=["sex"])

    if "oral_health_status" in df.columns:
        df["oral_health_status_encoded"] = df["oral_health_status"].map(
            CATEGORICAL_MAPPINGS["oral_health_status"]
        )
        df = df.drop(columns=["oral_health_status"])

    if "tartar_presence" in df.columns:
        df["tartar_presence_encoded"] = df["tartar_presence"].map(
            CATEGORICAL_MAPPINGS["tartar_presence"]
        )
        df = df.drop(columns=["tartar_presence"])

    return df


def preprocess_simple(data_dir: str = None, output_prefix: str = "simple"):
    """Run simplified preprocessing and save files."""
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = script_dir

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Save IDs separately
    ids = test_df[["patient_id"]]
    ids_output = os.path.join(data_dir, f"ids_{output_prefix}.csv")
    ids.to_csv(ids_output, index=False)
    print(f"Saved IDs to {ids_output}")

    # Drop ID column
    train_df_nod = train_df.drop(columns=["patient_id"])
    test_df_nod = test_df.drop(columns=["patient_id"])

    # Encode categorical columns
    train_processed = encode_categorical(train_df_nod)
    test_processed = encode_categorical(test_df_nod)

    # Save processed datasets
    train_out = os.path.join(data_dir, f"train_processed_{output_prefix}.csv")
    test_out = os.path.join(data_dir, f"test_processed_{output_prefix}.csv")
    train_processed.to_csv(train_out, index=False)
    test_processed.to_csv(test_out, index=False)

    print(f"Saved train features to {train_out}")
    print(f"Saved test features to {test_out}")
    print("Remaining columns:", train_processed.columns.tolist())


if __name__ == "__main__":
    preprocess_simple()

