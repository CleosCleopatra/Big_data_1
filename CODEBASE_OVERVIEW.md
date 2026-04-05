# Codebase Overview

## Repository Summary

This repository is a small course project centered on classifying handwritten digits from a reduced MNIST dataset. The codebase is mostly contained in a single Jupyter notebook, with the dataset stored as CSV files.

The repo contains:

- `project1Py.ipynb`: the main analysis notebook
- `mnist_large/images.csv`: flattened 28x28 grayscale digit images
- `mnist_large/labels.csv`: labels for each image
- `mnist_large.zip`: zipped copy of the dataset

## What The Project Does

The notebook walks through a typical machine learning workflow:

1. Load image and label data from CSV files.
2. Explore the data with pandas and NumPy.
3. Visualize random digit samples.
4. Build a basic classifier on a subset of digits with k-nearest neighbors.
5. Start Part 1 of the assignment:
   - reduce dimensionality with SVD
   - inspect class separation
   - split data into train/validation/test sets
   - repeatedly evaluate KNN across different hyperparameters
6. Leave placeholders for later work and for the course's Part 2 themes.

## Repository Structure

### `project1Py.ipynb`

This is the real application code. It mixes:

- assignment instructions
- exploratory data analysis
- visualization
- utility functions
- repeated train/validation/test experiments

The notebook is written as a linear workflow rather than as a package with modules.

### `mnist_large/images.csv`

This file stores the input features. Each row is one digit image flattened into 784 pixel values.

Conceptually:

- image size: 28 x 28
- features per image: 784
- one row = one sample

### `mnist_large/labels.csv`

This file stores the class label for each corresponding row in `images.csv`.

The notebook renames the default column `"0"` to `"label"` when using pandas.

### `mnist_large.zip`

This is a compressed copy of the same dataset folder. It is not part of the execution logic once the CSVs are already extracted.

## Notebook Walkthrough

## 1. Data Loading

The notebook begins by loading the CSV files with pandas:

- `PATHIM = "mnist_large//images.csv"`
- `PATHLB = "mnist_large//labels.csv"`

It reads both files, renames the label column, and prints the shapes of the raw data.

It also demonstrates class filtering by selecting only digits `0` and `9` with a boolean mask:

- `images_0_9`
- `labels_0_9`

This section is mainly exploratory and intended to help understand dataset shape and class distribution.

## 2. Exploratory Analysis With Pandas

The pandas-based exploration does two main things:

- checks class counts with `labels["label"].value_counts()`
- compares average pixel intensity across classes

To do that, the notebook:

1. joins image rows and labels into one DataFrame
2. computes `mean_intensity` per image
3. creates class-wise histograms using `pivot(...).hist(...)`

Purpose:

- get a quick sense of whether some digits differ in overall brightness
- show that even simple summary statistics can reveal class structure

## 3. Image Visualization With NumPy and Matplotlib

After the pandas section, the notebook converts the loaded tables into NumPy arrays:

- `images = np.array(images)`
- `labels = np.array(labels).ravel()`

It then:

- samples 9 random images
- reshapes each row into `28 x 28`
- plots them in a `3 x 3` grid

This is a sanity check that verifies:

- the CSV rows are valid images
- labels line up with the images
- the data looks like handwritten digits

## 4. Baseline Classification Example

The next section is a small, self-contained example using three classes:

- digits `1`, `5`, and `7`

Workflow:

1. filter data down to those three labels
2. split into train and test sets with `train_test_split(..., stratify=labels_sub)`
3. print per-class train/test counts
4. train `KNeighborsClassifier(n_neighbors=3)`
5. predict on the test set
6. display a confusion matrix

Why this section matters:

- it proves the basic classification setup works
- it introduces a supervised learning pipeline before the larger repeated experiment loop

## 5. Part 1: Assignment Setup

The notebook then switches from demonstration into the actual course assignment.

The assignment requires:

- dimension reduction
- at least four classifiers of different types
- tuning with cross-validation where needed
- comparison on a held-out test set

At the moment, only part of that requirement is implemented in code. The notebook mainly develops an SVD + KNN pipeline.

## 6. Commented-Out SVD Exploration

There are large triple-quoted blocks containing experimental code that is currently disabled.

That commented code includes:

- sampling 2000 images from the training set
- computing SVD on that sample
- plotting cumulative explained variance
- reconstructing a single image with different numbers of singular components

Purpose of that code:

- understand how many singular vectors are needed to preserve most structure
- visualize what low-rank image approximations look like

Because this code is wrapped in triple quotes, it does not execute.

## 7. Manual Train/Test Split

The notebook then manually creates a train/test split using NumPy:

- 2000 random samples for training
- all remaining samples for testing

Variables:

- `X_train_org`
- `y_train`
- `X_test_org`
- `y_test`

This split is used for the first SVD-based inspection of class separation.

## 8. SVD for Dimension Reduction

The code computes SVD on the training matrix:

- `U_train, S_train, vt_train = np.linalg.svd(X_train_org, full_matrices=False)`

It then visualizes:

- principal direction 1 vs 2
- principal direction 3 vs 4

using scatter plots colored by digit label.

Interpretation in the notebook:

- some digits separate reasonably well
- many classes overlap
- the first few components are not enough for full separation

This section is the notebook's main answer to the assignment question "Is the data well separated?"

## 9. Utility Functions

The notebook defines three reusable helper functions.

### `calc_reduced_pattern(U, S, vt, p_val, X_test_org, X_val_org)`

Purpose:

- project training, validation, and test data into a lower-dimensional SVD space

What it returns:

- projected training data using the first `p_val` components
- projected test data
- projected validation data

Implementation idea:

- training projection uses `U[:, :p_val] @ diag(S[:p_val])`
- validation and test projection use the first `p_val` rows of `vt`

This function is the bridge between SVD and classification.

### `visuale_error(x_axs, y_axs, x_name, y_name, title, results_val, ord_x, ord_y)`

Purpose:

- visualize model error distributions across hyperparameter settings

What it does:

- builds grouped boxplots from repeated experiment results
- colors each group by one hyperparameter
- labels the axes and legend

Despite the name, this is a plotting function for hyperparameter comparison.

Note:

- there is a large block of older heatmap code inside the function that is commented out
- `bp_list` is created but never used

### `compute_SVD(X, y, n)`

Purpose:

- create a fresh train/validation/test split for one experiment repetition
- center the training data
- compute SVD from the centered training set

What it returns:

- train inputs and labels
- centered validation inputs and labels
- centered test inputs and labels
- `U_train`, `S_train`, `vt_train`

Important implementation detail:

- the mean is computed only from the training data
- validation and test sets are centered using that same training mean

That is the correct pattern for avoiding information leakage from validation/test into training.

## 10. Repeated KNN Experiment

The main implemented experiment is a repeated search over:

- `k_vals = [1, 2, 5, 10, 25, 50]`
- `P_vals = [2, 3, 10, 25, 50, 100, 150, 200]`
- `B = 25` repeated random splits

For each repetition:

1. `compute_SVD(...)` creates train/validation/test sets.
2. `calc_reduced_pattern(...)` projects the data to `p_val` dimensions.
3. A KNN classifier is trained for each `k`.
4. Validation and test predictions are made.
5. Error rates are appended to:
   - `results_val`
   - `results_test`

At the end:

- both result lists are converted to NumPy arrays
- `visuale_error(...)` plots the validation error distributions

This is the core experimental pipeline in the repo.

## What Is Implemented vs Missing

### Implemented

- CSV-based data loading
- basic exploratory analysis
- random image visualization
- one baseline KNN example on three digits
- SVD-based dimensionality reduction
- repeated KNN tuning over number of neighbors and retained dimensions
- validation/test error collection
- separation plots in low-dimensional component space

### Present But Not Executed

- exploratory SVD reconstruction code inside triple-quoted blocks
- singular-value threshold plotting code inside triple-quoted blocks

### Not Yet Implemented

- comparison of at least four classifier families
- systematic cross-validation for multiple model types
- final model selection logic after "Take the best comb of k and P"
- full Part 2 theme implementation
- notebook refactoring into reusable modules or scripts

## Data Flow

The data flow through the notebook is:

1. Load CSV files with pandas.
2. Convert to NumPy arrays.
3. Create random train/validation/test splits.
4. Center the training data.
5. Compute SVD on centered training data.
6. Project train/validation/test sets onto the first `p` components.
7. Train KNN in reduced space.
8. Measure validation and test error.
9. Repeat the whole process across many random splits.

## Main Libraries Used

The notebook relies on:

- `pandas` for CSV loading and tabular inspection
- `numpy` for array operations, masking, random sampling, and SVD
- `matplotlib` for image plots, scatter plots, histograms, and boxplots
- `scikit-learn` for:
  - `KNeighborsClassifier`
  - `train_test_split`
  - `confusion_matrix`
  - `ConfusionMatrixDisplay`

There is also some unused or disabled import code in commented sections.

## Strengths Of The Current Code

- The workflow follows a sensible ML pipeline.
- Training mean centering is handled correctly in `compute_SVD`.
- Repeated random splits reduce the risk of drawing conclusions from a single lucky run.
- The notebook combines exploratory analysis with modeling, which is appropriate for a course project.

## Weaknesses And Technical Debt

### 1. Most logic lives in one notebook

This makes the project harder to:

- test
- reuse
- review
- run as a clean pipeline

### 2. Assignment scope is only partially implemented

The assignment asks for at least four classifiers, but the actual code mainly evaluates KNN.

### 3. Reproducibility is inconsistent

Some sections set random seeds, but others use random sampling without a fixed seed for each run.

### 4. There is duplicated or dead code

- commented blocks are large
- some variables are created and not used
- there are placeholder cells with no implementation

### 5. Manual splitting is less robust than sklearn utilities

Some splits are done manually with NumPy instead of using a consistent helper or sklearn splitter.

### 6. No project-level README or requirements file

A new reader does not immediately know:

- how to run the notebook
- which package versions are needed
- what parts are complete

## How To Read The Notebook Efficiently

If someone new opens this repo, the best reading order is:

1. Data loading and pandas exploration
2. NumPy image visualization
3. Baseline KNN example
4. SVD plots and helper functions
5. Repeated KNN experiment loop
6. Assignment text and placeholders for future work

## Suggested Next Refactor

If you want to turn this into a cleaner codebase, the next practical step would be to split the notebook into:

- `data_loading.py`
- `dimensionality_reduction.py`
- `models.py`
- `evaluation.py`
- `visualization.py`

Then keep the notebook only for presentation and interpretation.

## Bottom Line

This codebase is a notebook-driven MNIST classification project. Its implemented core is an SVD + KNN experimentation pipeline with repeated random splits, supported by basic data exploration and visualization. It is a good project scaffold for a statistics or machine learning assignment, but it is not yet a complete multi-model classification framework.
