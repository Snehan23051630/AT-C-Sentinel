"""Reproducible customer split and preprocessing fitted on training data only."""

import numpy as np
import pandas as pd


def load_saved_workflow(source="data set.csv", output="processed"):
    """Restore notebook DataFrames, raw histories and frozen training statistics."""
    from pathlib import Path

    output_dir = Path(output)
    state = {"output_dir": output_dir}
    with np.load(output_dir / "sgcc_split.npz", allow_pickle=False) as saved:
        dates = pd.DatetimeIndex(saved["dates"])
        for part in ("train", "test"):
            rows = saved[f"{part}_rows"]
            state[f"{part}_idx"] = rows
            state[f"X_{part}"] = pd.DataFrame(
                saved[f"X_{part}"], index=rows, columns=dates)
            state[f"y_{part}"] = pd.Series(
                saved[f"y_{part}"], index=rows, name="FLAG")
            state[f"{part}_ids"] = pd.Series(
                saved[f"{part}_ids"], index=rows, name="CONS_NO")
    stats = pd.read_csv(output_dir / "training_preprocessing.csv", index_col="date")
    stats.index = pd.to_datetime(stats.index)
    if not stats.index.equals(dates):
        raise ValueError("Saved statistics and consumption dates do not match")
    state["preprocessing_stats"] = stats
    print("Loaded cleaned data and training statistics. Reading raw histories...", flush=True)
    df = pd.read_csv(source)
    X = df.drop(columns=["CONS_NO", "FLAG"])
    X.columns = pd.to_datetime(X.columns)
    X = X.sort_index(axis=1)
    if not X.columns.equals(dates) or not df["CONS_NO"].is_unique:
        raise ValueError("Source data does not match saved split")
    all_rows = np.concatenate([state["train_idx"], state["test_idx"]])
    if not np.array_equal(np.sort(all_rows), np.arange(len(df))):
        raise ValueError("Saved split must cover every source customer exactly once")
    for part in ("train", "test"):
        rows = state[f"{part}_idx"]
        if (not np.array_equal(df.loc[rows, "CONS_NO"], state[f"{part}_ids"])
                or not np.array_equal(df.loc[rows, "FLAG"], state[f"y_{part}"])):
            raise ValueError("Source customer order/labels changed; rebuild the split")
        state[f"X_{part}_raw"] = X.loc[rows].copy()
    state.update(df=df, X=X, label=df["FLAG"])
    state["split_summary"] = pd.DataFrame({
        "customers": [len(state[f"y_{p}"]) for p in ("train", "test")],
        "theft_customers": [int(state[f"y_{p}"].sum()) for p in ("train", "test")],
        "theft_percent": [100 * state[f"y_{p}"].mean() for p in ("train", "test")],
    }, index=["train", "test"])
    print("Ready: cleaned data, raw histories, labels, IDs and statistics restored.", flush=True)
    return state


def stratified_split(labels, test_fraction=0.2, seed=42):
    """Shuffle within each class, reserve ~20%, then shuffle both partitions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("FLAG must contain both binary classes and no missing labels")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for value in sorted(labels.unique()):
        indices = rng.permutation(labels.index[labels == value].to_numpy())
        n_test = round(len(indices) * test_fraction)
        if not 0 < n_test < len(indices):
            raise ValueError("Each class needs enough customers for both partitions")
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return rng.permutation(train), rng.permutation(test)


def fit_preprocessing(train):
    """Daily median imputation and conservative upper-tail winsorization.

    Quantiles ignore missing values, so imputation does not distort the cap.
    Entirely missing training dates use the median of training daily medians
    for imputation and the largest available training daily cap for clipping.
    """
    observed = train.replace([np.inf, -np.inf], np.nan)
    medians = observed.median()
    caps = observed.quantile(0.999)
    if medians.isna().all():
        raise ValueError("Training data contains no finite readings")
    return pd.DataFrame({
        "median": medians.fillna(medians.median()),
        "upper_cap": caps.fillna(caps.max()),
    })


def transform(readings, statistics):
    """Apply frozen training statistics without estimating anything from input."""
    if not readings.columns.equals(statistics.index):
        raise ValueError("Reading dates must match the fitted preprocessing dates")
    return (readings.replace([np.inf, -np.inf], np.nan)
            .fillna(statistics["median"])
            .clip(upper=statistics["upper_cap"], axis=1))


def _longest_run(mask):
    """Longest consecutive True run per customer, in chronological columns."""
    current = np.zeros(len(mask), dtype=np.int64)
    longest = current.copy()
    for column in mask.T:
        current = np.where(column, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def extract_features(raw, cleaned):
    """Return 13 customer features while preserving the input row index.

    Inputs must share a unique DatetimeIndex in ascending order.
    Windows and runs count recorded date columns (absent dates are not inserted).
    Consumption summaries use finite, training-imputed/capped cleaned values;
    zero/missing features use raw values (infinities count as missing).
    std uses ddof=0; cv=std/abs(mean); peak_ratio=max/abs(mean).
    Undefined zero-denominator ratios are zero. zero_frac and miss_ratio
    use all date slots as denominator; missing values interrupt zero runs.
    mean_abs_change averages absolute consecutive-day changes (0 for one day).
    drop_ratio=max(0, 1-last90_mean/previous90_mean), bounded above by 1;
    it is zero when no previous window exists or its mean is nonpositive.
    Short histories use the available days in each window.
    """
    if not raw.index.equals(cleaned.index) or not raw.columns.equals(cleaned.columns):
        raise ValueError("Raw and cleaned rows/dates must match exactly")
    dates = raw.columns
    if (not isinstance(dates, pd.DatetimeIndex) or len(dates) == 0
            or dates.hasnans or not dates.is_unique
            or not dates.is_monotonic_increasing):
        raise ValueError("Features require sorted, unique date columns")
    values = cleaned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cleaned readings must all be finite")
    observed = raw.to_numpy(dtype=float)
    missing = ~np.isfinite(observed)
    zeros = observed == 0
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=0)
    maximum = values.max(axis=1)

    def safe_ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(mean),
                         where=denominator != 0)

    recent = values[:, -90:].mean(axis=1)
    drop = np.zeros_like(mean)
    if values.shape[1] > 90:
        previous = values[:, -180:-90].mean(axis=1)
        drop = np.where(previous > 0,
                        np.clip(1 - safe_ratio(recent, previous), 0, 1), 0)
    return pd.DataFrame({
        "mean": mean,
        "median": np.median(values, axis=1),
        "std": std,
        "max": maximum,
        "cv": safe_ratio(std, np.abs(mean)),
        "peak_ratio": safe_ratio(maximum, np.abs(mean)),
        "zero_frac": zeros.mean(axis=1),
        "longest_zero_run": _longest_run(zeros),
        "miss_ratio": missing.mean(axis=1),
        "longest_nan_run": _longest_run(missing),
        "miss_ratio_last90": missing[:, -90:].mean(axis=1),
        "mean_abs_change": (np.abs(np.diff(values, axis=1)).mean(axis=1)
                            if values.shape[1] > 1 else np.zeros_like(mean)),
        "drop_ratio": drop,
    }, index=raw.index)
