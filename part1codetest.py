import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PATHIM = "mnist_large/images.csv"
PATHLB = "mnist_large/labels.csv"


def load_data(image_path=PATHIM, label_path=PATHLB):
    images = pd.read_csv(image_path, sep=",", index_col=0)
    labels = pd.read_csv(label_path, sep=",", index_col=0)
    labels = labels.rename(columns={"0": "label"})
    X = images.to_numpy()
    y = labels["label"].to_numpy()
    return X, y


def get_models(random_state=42):
    models = {
        "knn": {
            "model": Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 9],
                "clf__weights": ["uniform", "distance"]
            }
        },
        "logistic": {
            "model": Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000))
            ]),
            "params": {
                "clf__C": [0.1, 1, 10]
            }
        },
        "nearest_centroid": {
            "model": Pipeline([
                ("scale", StandardScaler()),
                ("clf", NearestCentroid())
            ]),
            "params": {}
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1
            ),
            "params": {
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 3]
            }
        }
    }
    return models


def class_accuracy(y_true, y_pred):
    out = {}
    for digit in np.unique(y_true):
        mask = y_true == digit
        out[int(digit)] = np.mean(y_pred[mask] == y_true[mask])
    return out


def plot_pca_separation(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    pca_plot = PCA(n_components=4)
    train_scores_plot = pca_plot.fit_transform(X_train)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    scatter1 = ax[0].scatter(
        train_scores_plot[:, 0],
        train_scores_plot[:, 1],
        c=y_train,
        cmap="tab10",
        s=15,
        alpha=0.6
    )
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("Training data: PC1 vs PC2")
    ax[0].grid(True)
    plt.colorbar(scatter1, ax=ax[0], label="Digit")

    scatter2 = ax[1].scatter(
        train_scores_plot[:, 2],
        train_scores_plot[:, 3],
        c=y_train,
        cmap="tab10",
        s=15,
        alpha=0.6
    )
    ax[1].set_xlabel("PC3")
    ax[1].set_ylabel("PC4")
    ax[1].set_title("Training data: PC3 vs PC4")
    ax[1].grid(True)
    plt.colorbar(scatter2, ax=ax[1], label="Digit")

    plt.tight_layout()
    plt.show()

    print(
        "The data is only partly separated in the first few principal components. "
        "Some digits separate fairly well, but several classes still overlap. "
        "So PCA helps us see structure in the data, but the first few components "
        "are not enough for perfect classification."
    )


def evaluate_models(X, y, n_repeats=5, n_pcs=50, base_random_state=42):
    results = []

    for b in range(n_repeats):
        print(f"Repeat {b + 1} of {n_repeats}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=base_random_state + b,
            stratify=y
        )

        pca = PCA(n_components=n_pcs)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=base_random_state + b
        )

        models = get_models(random_state=base_random_state + b)

        for model_name in models:
            model_info = models[model_name]
            model = clone(model_info["model"])
            param_grid = model_info["params"]

            if len(param_grid) > 0:
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1
                )
                search.fit(X_train_pca, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_cv_score = search.best_score_
            else:
                best_model = model.fit(X_train_pca, y_train)
                best_params = {}
                best_cv_score = np.nan

            y_pred_test = best_model.predict(X_test_pca)
            test_acc = np.mean(y_pred_test == y_test)

            results.append({
                "repeat": b,
                "model": model_name,
                "cv_accuracy": best_cv_score,
                "test_accuracy": test_acc,
                "best_params": best_params,
                "class_accuracy": class_accuracy(y_test, y_pred_test)
            })

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df.groupby("model")
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_cv_accuracy=("cv_accuracy", "mean")
        )
        .reset_index()
        .sort_values("mean_test_accuracy", ascending=False)
    )

    return results_df, summary_df


def plot_model_comparison(results_df):
    ordered_models = (
        results_df.groupby("model")["test_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    plot_data = [
        results_df.loc[results_df["model"] == model_name, "test_accuracy"]
        for model_name in ordered_models
    ]

    plt.figure(figsize=(10, 5))
    plt.boxplot(plot_data, tick_labels=list(ordered_models))
    plt.xlabel("Model")
    plt.ylabel("Test accuracy")
    plt.title("Repeated test accuracy for the different classifiers")
    plt.xticks(rotation=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def fit_final_model(X, y, best_model_name, best_params, n_pcs=50, random_state=123):
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    pca_final = PCA(n_components=n_pcs)
    X_train_final_pca = pca_final.fit_transform(X_train_final)
    X_test_final_pca = pca_final.transform(X_test_final)

    models_final = get_models(random_state=random_state)
    final_model = clone(models_final[best_model_name]["model"])
    final_model.set_params(**best_params)
    final_model.fit(X_train_final_pca, y_train_final)

    y_pred_final = final_model.predict(X_test_final_pca)
    final_test_accuracy = np.mean(y_pred_final == y_test_final)

    cm = confusion_matrix(y_test_final, y_pred_final, labels=np.unique(y))
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(
        cmap="Greens",
        xticks_rotation=45
    )
    plt.tight_layout()
    plt.show()

    return final_model, final_test_accuracy


def main():
    X, y = load_data()

    print(f"Images shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print()

    plot_pca_separation(X, y, random_state=42)

    results_df, summary_df = evaluate_models(
        X,
        y,
        n_repeats=5,
        n_pcs=50,
        base_random_state=42
    )

    print()
    print("Summary of repeated results:")
    print(summary_df.round(4))
    print()

    plot_model_comparison(results_df)

    print("Best runs:")
    print(
        results_df.sort_values(
            ["test_accuracy", "cv_accuracy"],
            ascending=False
        ).head(10)
    )
    print()

    best_model_name = summary_df.iloc[0]["model"]
    best_rows = results_df[results_df["model"] == best_model_name].sort_values(
        ["test_accuracy", "cv_accuracy"],
        ascending=False
    )
    best_params = best_rows.iloc[0]["best_params"]

    print("Best model:", best_model_name)
    print("Best parameters:", best_params)
    print("Mean test accuracy:", round(summary_df.iloc[0]["mean_test_accuracy"], 4))
    print()

    final_model, final_test_accuracy = fit_final_model(
        X,
        y,
        best_model_name,
        best_params,
        n_pcs=50,
        random_state=123
    )

    print("Final held-out test accuracy:", round(final_test_accuracy, 4))
    print()
    print(
        "Part 1 is complete. We did dimension reduction with PCA, compared four different "
        "classifiers, used cross-validation for tuning, and compared the models on repeated "
        "test sets."
    )


if __name__ == "__main__":
    main()
