"""Training-only baseline selection with a single saved holdout evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RULES = {"drop": "drop_ratio", "missing_data": "miss_ratio"}


def average_precision(labels, scores):
    """Non-interpolated PR-AUC (average precision), grouping tied scores."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or labels.shape != scores.shape or len(labels) == 0
            or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(scores).all() or labels.sum() == 0):
        raise ValueError("Expected aligned binary labels with positives and finite scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    true_positives = np.cumsum(labels[order])
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    precision = true_positives[ends] / (ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives[ends]]) / labels.sum()
    return float(np.sum(precision * recall_increments))


def validate_features(frame):
    if (frame.empty or frame["CONS_NO"].isna().any()
            or not frame["CONS_NO"].is_unique
            or set(frame["FLAG"].unique()) != {0, 1}):
        raise ValueError("Expected unique customer IDs and both binary FLAG classes")
    scores = frame[list(RULES.values())].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Baseline ratios must be finite and in [0, 1]")


def select_baseline(train):
    """Freeze score directions and choose highest training AP; ties favor chance."""
    validate_features(train)
    metrics = {"chance": float(train["FLAG"].mean())}
    metrics.update({name: average_precision(train["FLAG"], train[column])
                    for name, column in RULES.items()})
    return {
        "metric": "average_precision (non-interpolated PR-AUC)",
        "scores": {"chance": "analytical prevalence reference", **RULES},
        "direction": "higher is riskier; fixed before test evaluation",
        "threshold": None,
        "selection": "highest training PR-AUC; ties favor listed order",
        "train_pr_auc": metrics,
        "selected_baseline": max(metrics, key=metrics.get),
    }


def run_baselines(output="processed"):
    """Persist the training decision before opening the held-out feature table.

    Existing artifacts are never overwritten: subsequent notebook runs should
    display baseline_results.csv, not repeatedly evaluate the holdout.
    """
    output_dir = Path(output)
    selection_path = output_dir / "baseline_selection.json"
    results_path = output_dir / "baseline_results.csv"
    report_path = output_dir / "baseline_report.md"
    if any(path.exists() for path in (selection_path, results_path, report_path)):
        raise FileExistsError("Baseline artifacts already exist; read the saved results")
    train = pd.read_csv(output_dir / "train_features.csv", dtype={"CONS_NO": str})
    selection = select_baseline(train)
    with selection_path.open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    test = pd.read_csv(output_dir / "test_features.csv", dtype={"CONS_NO": str})
    validate_features(test)
    if not set(train["CONS_NO"]).isdisjoint(test["CONS_NO"]):
        raise ValueError("Training and test customers must be disjoint")
    prevalence = float(test["FLAG"].mean())
    test_metrics = {"chance": prevalence}
    test_metrics.update({name: average_precision(test["FLAG"], test[column])
                         for name, column in RULES.items()})
    results = pd.DataFrame([
        {"baseline": name, "train_pr_auc": train_score,
         "test_pr_auc": test_metrics[name],
         "test_lift_over_chance": test_metrics[name] / prevalence,
         "selected_on_train": name == selection["selected_baseline"]}
        for name, train_score in selection["train_pr_auc"].items()
    ])
    results.to_csv(results_path, index=False, mode="x")
    missing_ap = test_metrics["missing_data"]
    comparison = "above" if missing_ap > prevalence else "at or below"
    report = (
        "# Baseline benchmark\n\n"
        "Rules and score directions were fixed in advance. The baseline with the "
        "highest training AP was selected and saved before loading test data. "
        "All three baselines were then evaluated in one holdout pass, without "
        "test-based selection or tuning. No threshold is needed for ranking AP.\n\n"
        f"Training-selected baseline: **{selection['selected_baseline']}**.\n\n"
        "| Baseline | Train PR-AUC | Test PR-AUC | Test / chance |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    for row in results.itertuples(index=False):
        report += (f"| {row.baseline} | {row.train_pr_auc:.6f} | "
                   f"{row.test_pr_auc:.6f} | {row.test_lift_over_chance:.2f} |\n")
    report += (
        "\nPR-AUC here means non-interpolated average precision with tied scores "
        "grouped, not trapezoidal integration. Chance is the analytical "
        "prevalence reference, not a sampled random permutation: finite random "
        "rankings fluctuate around this reference.\n\n"
        f"Missingness-only AP is {missing_ap:.6f}, {missing_ap / prevalence:.2f}x "
        f"chance ({comparison} the prevalence reference). "
        "This measures how well a pure data-collection artifact can rank theft "
        "labels. Given the January 2016 collection break, above-chance performance "
        "is an artifact warning, not evidence of theft behavior or proof that "
        "the break caused the association.\n\n"
        "Drop uses the existing cleaned-history drop_ratio (last 90 versus "
        "preceding 90 recorded days); missingness uses raw-history miss_ratio. "
        "Preprocessing statistics were fitted on training customers only. "
        "These are fixed numbers for a model to beat; do not tune against test results.\n"
    )
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    return results


if __name__ == "__main__":
    print(run_baselines().to_string(index=False))
