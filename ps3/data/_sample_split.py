"""Deterministic, ID-based train/test assignment."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


def _validate_inputs(df: pd.DataFrame, id_columns, training_frac: float) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if isinstance(id_columns, str):
        columns = [id_columns]
    else:
        try:
            columns = list(id_columns)
        except TypeError as exc:
            raise TypeError("id_columns must be a column name or a sequence of names") from exc

    if not columns:
        raise ValueError("id_columns must contain at least one column name")
    if any(not isinstance(column, str) for column in columns):
        raise TypeError("all entries in id_columns must be strings")
    if len(set(columns)) != len(columns):
        raise ValueError("id_columns must not contain duplicate column names")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"ID column(s) not found in df: {missing}")

    if isinstance(training_frac, bool) or not isinstance(training_frac, numbers.Real):
        raise TypeError("training_frac must be a number between 0 and 1")
    if not np.isfinite(training_frac) or not 0 <= training_frac <= 1:
        raise ValueError("training_frac must be between 0 and 1 inclusive")

    return columns


def create_sample_column(
    df: pd.DataFrame, id_columns, training_frac: float = 0.8
) -> pd.DataFrame:
    """Return a copy of ``df`` with a stable ``sample`` column.

    Each unique ID (or combination of IDs) is mapped to a deterministic
    64-bit hash bucket. Buckets below ``training_frac * 2**64`` are assigned
    to ``"train"``; the rest are assigned to ``"test"``. The assignment is
    independent of row order, dataframe index and the other records present.

    ``training_frac`` is therefore the expected training proportion, not a
    guarantee of an exact row-level proportion. All rows sharing an ID key
    receive the same assignment.
    """
    columns = _validate_inputs(df, id_columns, training_frac)

    result = df.copy()
    # pandas' stable value hashing avoids Python's process-randomised hash().
    # index=False ensures that reordering rows or changing the index has no effect.
    row_hashes = pd.util.hash_pandas_object(
        result.loc[:, columns], index=False, categorize=True
    ).to_numpy(dtype="uint64")

    threshold = int(float(training_frac) * (1 << 64))
    is_train = row_hashes < threshold
    result["sample"] = np.where(is_train, "train", "test")
    return result


def create_sample_split(df: pd.DataFrame, id_column, training_frac: float = 0.8):
    """Backward-compatible alias for :func:`create_sample_column`.

    ``id_column`` may be either one column name or a sequence of key columns.
    """
    return create_sample_column(df, id_columns=id_column, training_frac=training_frac)
