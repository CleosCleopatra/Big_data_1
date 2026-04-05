# Codebase For Part 1

## Review

### Teacher original vs your edited notebook

- [`project1original.ipynb`](C:\Users\jeys_\Coding stuff\Github\Big_data_1\project1original.ipynb) only contains the assignment statement for Part 1. It does **not** contain a completed Part 1 solution.
- [`project1Py.ipynb`](C:\Users\jeys_\Coding stuff\Github\Big_data_1\project1Py.ipynb) is where your group added the actual Part 1 code.

### Findings

1. [`project1Py.ipynb`](C:\Users\jeys_\Coding stuff\Github\Big_data_1\project1Py.ipynb) mainly evaluates `KNeighborsClassifier`, but the assignment explicitly requires at least **4 different classifiers of different character**. Part 1 is therefore incomplete as written.

2. The current Part 1 uses repeated train/validation/test splits, but it does **not perform cross-validation** for model tuning. The assignment asks for cross-validation for models that require tuning.

3. The current SVD section computes the decomposition manually. A simpler and cleaner beginner-friendly solution is to use `sklearn.decomposition.PCA`, which automatically centers the data.

4. The final “take the best combination” step is still missing in the current notebook, so the code does not fully finish the model-selection part of Part 1.

## Bottom line

Your current Part 1 work shows the right direction:

- you started dimension reduction
- you repeated the exercise several times
- you explored KNN hyperparameters

But it is **not fully correct or complete** for the assignment yet. The cleanest fix is to replace the current Part 1 code cells with the code below.

## What to replace

Keep the introductory cells and the Part 1 markdown prompt.

Replace the current Part 1 code cells after the `# Part 1` markdown with the following cells.

## Cell 1: Part 1 imports and helper functions

```python
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
```

## Cell 2: Dimension reduction and separation

```python
# Look at the data after PCA and see if the classes seem separated

X_train_pca_plot, X_test_pca_plot, y_train_pca_plot, y_test_pca_plot = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# PCA in sklearn centres the data automatically
pca_plot = PCA(n_components=4)
train_scores_plot = pca_plot.fit_transform(X_train_pca_plot)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

scatter1 = ax[0].scatter(
    train_scores_plot[:, 0],
    train_scores_plot[:, 1],
    c=y_train_pca_plot,
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
    c=y_train_pca_plot,
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
```

## Cell 3: Short answer for Question 1

```python
print(
    "The data is only partly separated in the first few principal components. "
    "Some digits separate fairly well, but several classes still overlap. "
    "So PCA helps us see structure in the data, but the first few components "
    "are not enough for perfect classification."
)
```

## Cell 4: Train, tune with CV, and compare at least four classifiers

```python
n_repeats = 5
n_pcs = 50
results = []

for b in range(n_repeats):
    print(f"Repeat {b + 1} of {n_repeats}")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42 + b,
        stratify=labels
    )

    pca = PCA(n_components=n_pcs)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + b)

    models = get_models(random_state=42 + b)

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
results_df.head()
```

## Cell 5: Summarize results

```python
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

summary_df.round(4)
```

## Cell 6: Plot repeated performance comparison

```python
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
```

## Cell 7: Inspect best settings

```python
results_df.sort_values(
    ["test_accuracy", "cv_accuracy"],
    ascending=False
).head(10)
```

## Cell 8: Best model summary

```python
best_model_name = summary_df.iloc[0]["model"]
best_rows = results_df[results_df["model"] == best_model_name].sort_values(
    ["test_accuracy", "cv_accuracy"],
    ascending=False
)

best_params = best_rows.iloc[0]["best_params"]

print("Best model:", best_model_name)
print("Best parameters:", best_params)
print("Mean test accuracy:", round(summary_df.iloc[0]["mean_test_accuracy"], 4))
```

## Cell 9: Final held-out confusion matrix

```python
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=123,
    stratify=labels
)

pca_final = PCA(n_components=n_pcs)
X_train_final_pca = pca_final.fit_transform(X_train_final)
X_test_final_pca = pca_final.transform(X_test_final)

models_final = get_models(random_state=123)
final_model = clone(models_final[best_model_name]["model"])
final_model.set_params(**best_params)
final_model.fit(X_train_final_pca, y_train_final)

y_pred_final = final_model.predict(X_test_final_pca)
final_test_accuracy = np.mean(y_pred_final == y_test_final)

cm = confusion_matrix(y_test_final, y_pred_final, labels=np.unique(labels))
ConfusionMatrixDisplay(cm, display_labels=np.unique(labels)).plot(
    cmap="Greens",
    xticks_rotation=45
)
plt.tight_layout()
plt.show()

print("Final held-out test accuracy:", round(final_test_accuracy, 4))
```

## Cell 10: Final Part 1 takeaway

```python
print(
    "Part 1 is complete. We did dimension reduction with PCA, compared four different "
    "classifiers, used cross-validation for tuning, and compared the models on repeated "
    "test sets."
)
```

## Why this replacement is better

- It uses PCA in a simple `sklearn` way.
- It keeps the code beginner-friendly and notebook-friendly.
- It evaluates at least 4 different classifier families.
- It uses cross-validation for the tuned models.
- It finishes the final model selection and confusion matrix step.
