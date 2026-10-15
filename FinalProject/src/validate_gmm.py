import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


def validate_gmm(features, n_components=3):

    print("\n" + "=" * 60)
    print("GMM GROUP VALIDATION")
    print("=" * 60)

    data = features.copy()

    feature_columns = [
        "start_height_z",
        "end_height_z",
        "height_z_change",
        "height_z_rate",
        "start_weight_z",
        "end_weight_z",
        "weight_z_change",
        "weight_z_rate",
    ]

    X = data[feature_columns].copy()


    valid = X.notna().all(axis=1)

    data = data.loc[valid].copy()
    X = X.loc[valid].copy()

    print(f"\nChildren used for GMM validation: {len(data):,}")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # fit gmm
    # ---------------------------------------------------------

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        n_init=10
    )

    labels = gmm.fit_predict(X_scaled)

    data["growth_group"] = labels


    print("\n=== GROUP DISTRIBUTION ===")

    counts = (
        data["growth_group"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts / len(data) * 100
    ).round(2)

    distribution = pd.DataFrame({
        "count": counts,
        "percent": percentages
    })

    print(distribution)

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    print("\n=== SILHOUETTE SCORE ===")
    print(f"Silhouette score: {silhouette:.4f}")

    # ---------------------------------------------------------
    # group means
    # ---------------------------------------------------------

    print("\n=== MEAN FEATURES BY GROUP ===")

    group_means = (
        data.groupby("growth_group")[feature_columns]
        .mean()
        .round(3)
    )

    print(group_means)

    # ---------------------------------------------------------
    # group median and means
    # ---------------------------------------------------------

    print("\n=== MEDIAN FEATURES BY GROUP ===")

    group_medians = (
        data.groupby("growth_group")[feature_columns]
        .median()
        .round(3)
    )

    print(group_medians)


    standardized_df = pd.DataFrame(
        X_scaled,
        columns=feature_columns,
        index=data.index
    )

    standardized_df["growth_group"] = labels

    print("\n=== STANDARDIZED GROUP MEANS ===")

    standardized_means = (
        standardized_df
        .groupby("growth_group")
        .mean()
        .round(3)
    )

    print(standardized_means)

    # ---------------------------------------------------------
    # gmm prob. 
    # ---------------------------------------------------------

    probabilities = gmm.predict_proba(X_scaled)

    confidence = probabilities.max(axis=1)

    data["cluster_confidence"] = confidence

    print("\n=== GMM ASSIGNMENT CONFIDENCE ===")

    print(
        pd.Series(confidence).describe()
    )

    print(
        f"\nMean assignment confidence: "
        f"{confidence.mean():.3f}"
    )

    print(
        f"Assignments >= 0.80: "
        f"{(confidence >= 0.80).sum():,} "
        f"({(confidence >= 0.80).mean() * 100:.1f}%)"
    )

    print(
        f"Assignments >= 0.90: "
        f"{(confidence >= 0.90).sum():,} "
        f"({(confidence >= 0.90).mean() * 100:.1f}%)"
    )

    # ---------------------------------------------------------
    # stability
    # ---------------------------------------------------------

    print("\n=== GMM STABILITY ===")

    reference_labels = labels

    seeds = [
        1,
        7,
        21,
        42,
        100
    ]

    stability_results = []

    for seed in seeds:

        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
            n_init=10
        )

        new_labels = model.fit_predict(X_scaled)

        ari = adjusted_rand_score(
            reference_labels,
            new_labels
        )

        stability_results.append({
            "seed": seed,
            "ARI_vs_reference": ari,
            "converged": model.converged_,
            "iterations": model.n_iter_
        })

    stability = pd.DataFrame(
        stability_results
    )

    print(stability.to_string(index=False))

    # ---------------------------------------------------------
    # AIC BIC iterations
    # ---------------------------------------------------------

    print("\n=== MODEL INFORMATION ===")

    print(f"Components: {n_components}")
    print(f"Converged: {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    print(f"BIC: {gmm.bic(X_scaled):.2f}")
    print(f"AIC: {gmm.aic(X_scaled):.2f}")

    return (
        data,
        gmm,
        scaler,
        feature_columns
    )