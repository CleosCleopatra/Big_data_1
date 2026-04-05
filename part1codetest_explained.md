# part1codetest explained

This file explains how [`part1codetest.py`](C:\Users\jeys_\Coding stuff\Github\Big_data_1\part1codetest.py) works and how each part matches the Part 1 assignment in [`project1original.ipynb`](C:\Users\jeys_\Coding stuff\Github\Big_data_1\project1original.ipynb).

## Teacher's Part 1 assignment

The teacher asks you to do three main things in Part 1:

1. Perform dimension reduction on the data set and discuss whether the data is well separated.
2. Train classifiers using the training data, and use cross-validation to tune models that need tuning.
3. Compare the performance of different classifiers on a test set.

The assignment also says the exercise should be repeated so that conclusions are not based on one random split.

## What the script does

`part1codetest.py` is a complete Part 1 pipeline for the `mnist_large` data set.

It does the following:

1. Loads the image data and labels.
2. Uses PCA to reduce the dimension of the image data.
3. Plots the first principal components to inspect class separation.
4. Trains four different classifiers.
5. Uses cross-validation to tune the models that need tuning.
6. Repeats the train/test experiment several times.
7. Summarizes the results.
8. Selects the best model.
9. Fits the best model again on a fresh split and shows a confusion matrix.

That means the script covers the full Part 1 task.

## How the code is built

## 1. Loading the data

The function `load_data()` reads:

- `mnist_large/images.csv`
- `mnist_large/labels.csv`

and turns them into:

- `X`: the input matrix with pixel values
- `y`: the digit labels

This matches the teacher’s setup, where the project is based on the MNIST subset.

## 2. Creating the classifiers

The function `get_models()` creates the four classifiers used in Part 1:

- `knn`
- `logistic`
- `nearest_centroid`
- `random_forest`

This is important because the assignment says you should explore at least 4 different classifiers of different character.

Why these four make sense:

- `kNN` is a local, distance-based classifier.
- `LogisticRegression` is a linear parametric classifier.
- `NearestCentroid` is a simple prototype-based classifier.
- `RandomForestClassifier` is a tree-based nonlinear classifier.

So they are not all the same type of method.

## 3. Dimension reduction with PCA

The function `plot_pca_separation()` handles the first assignment question.

It:

1. Splits the data into training and test sets.
2. Fits `PCA(n_components=4)` on the training data.
3. Projects the training data into the first four principal components.
4. Plots:
   - PC1 vs PC2
   - PC3 vs PC4

This helps answer:

"Is the data well separated?"

The script then prints a short written conclusion saying that some digits separate reasonably well, but several classes still overlap.

This directly answers the first question in the assignment.

## 4. Why PCA is fitted on training data only

Inside both `plot_pca_separation()` and `evaluate_models()`, PCA is fitted on the training data only.

That is important because you should not use test information when building the model.  
If PCA were fitted on all the data first, information from the test set would leak into the training process.

This follows the course guidance about splitting the data appropriately before starting.

## 5. Repeated train/test splits

The function `evaluate_models()` repeats the whole Part 1 experiment several times.

It uses:

- `n_repeats=5`

For each repeat, it:

1. Splits the data into training and test sets.
2. Fits PCA on the training set.
3. Transforms both training and test data using the same PCA model.
4. Trains and tunes the classifiers.
5. Stores the test accuracy.

This matches the teacher’s instruction that one single run has limited value and that the task should be repeated.

## 6. Cross-validation for tuning

The script uses:

- `StratifiedKFold`
- `GridSearchCV`

This is the part that answers the second assignment question:

"Train the classifiers using the training data, and perform cross-validation to tune the models that require tuning."

How it works:

- For each repeated split, cross-validation is done only on the training set.
- The tuning grid depends on the classifier.

Examples:

- `kNN`: tunes number of neighbors and weighting scheme
- `LogisticRegression`: tunes `C`
- `RandomForest`: tunes `max_depth` and `min_samples_leaf`
- `NearestCentroid`: has no tuning grid in this version, so it is just fitted directly

Using `StratifiedKFold` also helps keep class proportions balanced in the folds.

## 7. Comparing the models

The script stores the result from every repeat in `results_df`.

That table includes:

- repeat number
- model name
- cross-validation accuracy
- test accuracy
- best parameters
- class-specific accuracy

Then it creates `summary_df`, which shows:

- mean test accuracy
- standard deviation of test accuracy
- mean cross-validation accuracy

This directly answers the third assignment question:

"Compare the performance of the different classifiers on a test set."

The comparison is not based on one split only. It is based on repeated test accuracy across several runs.

## 8. Boxplot of model performance

The function `plot_model_comparison()` creates a boxplot of repeated test accuracy for each classifier.

This is useful because it shows not only average performance, but also variation across splits.

So instead of saying:

- model A got one good result

you can say:

- model A usually performs better
- model B is less stable

That makes the comparison more informative for your report or slides.

## 9. Selecting the best model

After the repeated experiments, the script picks the best model using:

- highest mean test accuracy in `summary_df`

Then it looks at the best run for that model and extracts:

- `best_model_name`
- `best_params`

This completes the missing “choose the best model” step that your earlier notebook version had not finished.

## 10. Final confusion matrix

The function `fit_final_model()` does one final clean evaluation.

It:

1. Creates a fresh train/test split.
2. Fits PCA on the training data.
3. Builds the chosen best model using the selected parameters.
4. Predicts the test labels.
5. Computes the test accuracy.
6. Shows a confusion matrix.

The confusion matrix is helpful because it shows:

- which digits are classified well
- which digits get mixed up with each other

This gives more detail than overall accuracy alone.

## 11. Why this script matches the assignment well

This script is a good Part 1 solution because it does all the required parts:

- dimension reduction: yes, with PCA
- discussion of separation: yes, from the PCA plots and text
- at least 4 classifiers: yes
- cross-validation for tuning: yes
- test set comparison: yes
- repeated experiment: yes

So the script is not just code that runs. It is structured to answer the actual questions in the assignment.

## 12. How this relates to your notebook

`part1codetest.py` is basically the same Part 1 logic that should appear in the notebook, but written as one full script instead of separate cells.

That means:

- the notebook version is easier to present step by step
- the script version is easier to run all at once

Both are solving the same Part 1 task.

## 13. Short summary you can use in your report

If you want a short explanation for slides or notes, you can say:

> In Part 1, we used PCA to inspect the structure of the MNIST digit data and found that the first few components show some separation, but not perfect class separation. We then compared four different classifiers using repeated train/test splits and cross-validation on the training data. Finally, we selected the best-performing model and evaluated it on a held-out test set using both accuracy and a confusion matrix.
