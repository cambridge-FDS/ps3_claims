import hashlib

import numpy as np
import pandas as pd


# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(
    df: pd.DataFrame, id_column: str | list[str], training_frac: float = 0.9
) -> pd.DataFrame:
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    df_copy = df.copy()

    # Combine multiple ID columns into a single ID string.
    cols = [id_column] if isinstance(id_column, str) else id_column
    combined_ids = df_copy[cols].astype(str).agg("_".join, axis=1)

    # Extract random integers (0–99) from the ID string using hash
    hash_ints = combined_ids.apply(
        lambda x: int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16)
    )
    buckets = hash_ints % 100

    # Assign train/test labels
    threshold = int(training_frac * 100)
    df_copy["sample"] = np.where(buckets < threshold, "train", "test")

    return df_copy
