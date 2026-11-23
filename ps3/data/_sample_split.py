
"""Deterministic, ID-based train/test assignment."""

from __future__ import annotations

import json
import numbers

import numpy as np
import pandas as pd


def _validate_inputs(
    df: pd.DataFrame, id_columns, training_frac: float
) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if isinstance(id_columns, str):
        columns = [id_columns]
    else:
        try:
            columns = list(id_columns)
        except TypeError as exc:
            raise TypeError(
                "id_columns must be a column name or a sequence of names"
            ) from exc

    if not columns:
        raise ValueError("id_columns must contain at least one column name")

    if any(not isinstance(column, str) for column in columns):
        raise TypeError("all entries in id_columns must be strings")

    if len(set(columns)) != len(columns):
        raise ValueError("id_columns must not contain duplicate column names")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"ID column(s) not found in df: {missing}")

    if isinstance(training_frac, bool) or not isinstance(
        training_frac, numbers.Real
    ):
        raise TypeError("training_frac must be a number between 0 and 1")

    if not np.isfinite(training_frac) or not 0 <= training_frac <= 1:
        raise ValueError("training_frac must be between 0 and 1 inclusive")

    return columns


def _normalise_id(value) -> str:
    """Return a stable representation of a supported scalar ID.

    Integral floats and integers share a representation, so pandas dtype
    changes such as int -> float do not change an ID's assignment.
    Strings remain distinct from numeric IDs.
    """
    if pd.isna(value):
        raise ValueError("ID columns must not contain missing values")

    if isinstance(value, (bool, np.bool_)):
        return json.dumps(["bool", bool(value)])

    if isinstance(value, numbers.Integral):
        return json.dumps(["number", str(int(value))])

    if isinstance(value, numbers.Real):
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValueError("ID columns must not contain infinite values")

        if numeric_value.is_integer():
            return json.dumps(["number", str(int(numeric_value))])

        return json.dumps(["float", numeric_value.hex()])

    if isinstance(value, str):
        return json.dumps(["string", value], ensure_ascii=False)

    raise TypeError(
        f"Unsupported ID value type: {type(value).__name__}. "
        "Use strings, integers, booleans, or finite floats."
    )


def create_sample_column(
    df: pd.DataFrame, id_columns, training_frac: float = 0.8
) -> pd.DataFrame:
    """Return a copy of ``df`` with a stable ``sample`` column.

    Each unique ID (or combination of IDs) is mapped to a deterministic
    64-bit hash bucket. Buckets below ``training_frac * 2**64`` are assigned
    to ``"train"``; the rest are assigned to ``"test"``.

    Integer IDs and equivalent integral-float IDs receive the same
    assignment. String IDs remain distinct from numeric IDs.

    The assignment is independent of row order, dataframe index and the
    other records present. ``training_frac`` is the expected training
    proportion, not a guarantee of an exact row-level proportion.
    All rows sharing an ID key receive the same assignment.
    """
    columns = _validate_inputs(df, id_columns, training_frac)

    result = df.copy()

    # Canonicalise each ID column before hashing. This prevents pandas dtype
    # inference (for example, int -> float after appending a row) from
    # changing the hash of an existing ID.
    normalised = result.loc[:, columns].copy()

    for column in columns:
        normalised[column] = normalised[column].map(_normalise_id)

    row_hashes = pd.util.hash_pandas_object(
        normalised, index=False, categorize=True
    ).to_numpy(dtype="uint64")

    threshold = int(float(training_frac) * (1 << 64))
    is_train = row_hashes < threshold

    result["sample"] = np.where(is_train, "train", "test")
    return result


def create_sample_split(
    df: pd.DataFrame, id_column, training_frac: float = 0.8
):
    """Backward-compatible alias for :func:`create_sample_column`."""
    return create_sample_column(
        df, id_columns=id_column, training_frac=training_frac
    )
