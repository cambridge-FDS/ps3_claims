import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
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

    # 1. ID（整数）を文字列にして md5 → 安定した巨大整数へ変換
    hash_ints = df[id_column].astype(str).apply(
        lambda x: int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16)
    )

    # 2. 0〜99 の bucket に落とす（決定論的 split）
    buckets = hash_ints % 100

    # 3. training_frac に応じた閾値で train/test を決定
    threshold = int(training_frac * 100)
    df["sample"] = np.where(buckets < threshold, "train", "test")

    return df
