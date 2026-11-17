
# Weather-Up: Logistic Regression Experiments

## 1. Project Overview

The goal of this project is to predict `Outfit_Code` using weather-related data.

The dataset used is `Weather_Dataset_with_Outfit.csv`, containing over 8,000 observations.

This session focused on developing and improving a Logistic Regression classifier using Scikit-learn. Several preprocessing techniques, regularization methods, and class-weighting strategies were tested to improve classification performance.

## 2. Dataset Preparation

The dataset was loaded using Pandas:

```python
df = pd.read_csv("data/Weather_Dataset_with_Outfit.csv")
```

The dataset was initially divided into training and testing portions:

```python
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=RANDOM_STATE
)
```

The training portion was then divided again:

```python
X_train_no_proc, X_test_no_proc, y_train_no_proc, y_test_no_proc = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)
```

This resulted in the following approximate dataset distribution:

- Training: 56%
- Validation (cross-validation): 14%
- Final testing: 30%

The target variable was `Outfit_Code`, while the remaining columns were used as model features.

## 3. Data Preprocessing

A Scikit-learn `ColumnTransformer` was used to apply different preprocessing techniques to numerical and categorical features.

### Numerical Features

The following features were processed numerically:

- Time
- Temp_C
- Dew Point Temp_C
- Rel Hum_%
- Wind Speed_km/h
- Visibility_km
- Press_kPa

The numerical pipeline used:

1. `SimpleImputer(strategy="median")` to replace missing values.
2. `StandardScaler()` to standardize numerical features.

### Categorical Features

The `Weather` column was initially processed using `OrdinalEncoder`.

However, this was changed to `OneHotEncoder` because weather conditions are categorical rather than inherently ordered.

The final categorical pipeline used:

```python
text_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])
```

This allows the model to learn separate relationships for individual weather categories without introducing artificial numerical ordering.

## 4. Logistic Regression Experiments

Several Logistic Regression configurations were tested.

### Experiment 1: Initial Logistic Regression

The first model used the default Logistic Regression configuration.

A `RandomizedSearchCV` was used to investigate different regularization parameters.

The initial model produced convergence warnings because the default iteration limit was reached.

**Initial test accuracy: 85.77%**

### Experiment 2: Elastic Net Regularization

The next experiment introduced Elastic Net regularization by searching over:

- `C`
- `l1_ratio`

The `saga` solver was selected because it supports Elastic Net regularization.

The model was configured with:

```python
LogisticRegression(
    solver="saga",
    max_iter=3000,
    random_state=RANDOM_STATE
)
```

The parameter search was reduced to 30 candidates with 3-fold cross-validation to reduce computational time.

**Test accuracy: 88.46%**

Some convergence warnings remained.

### Experiment 3: L2 Regularization

The next experiment switched to the `lbfgs` solver and removed the `l1_ratio` parameter.

This allowed the model to use L2 regularization.

The parameter search was:

```python
logistic_params = {
    'C': np.logspace(-3, 3, 30)
}
```

The model used:

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE
)
```

**Test accuracy: 90.16%**

Convergence warnings were still observed.

### Experiment 4: One-Hot Encoding and Adjusted Regularization Search

The categorical preprocessing was improved by replacing `OrdinalEncoder` with `OneHotEncoder`.

The regularization search was also narrowed:

```python
logistic_params = {
    'C': np.logspace(-2, 1, 20)
}
```

The model was configured with:

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=5000,
    random_state=RANDOM_STATE
)
```

**Test accuracy: 93.98%**

This was the best-performing standard Logistic Regression configuration tested.

The improvement suggests that One-Hot Encoding was beneficial for representing the categorical weather feature.

## 5. Improving Cross-Validation

An important improvement was made to the overall model architecture.

Initially, preprocessing was performed before cross-validation:

```python
X_processed = preprocessor.fit_transform(X_train_no_proc)
```

This meant that preprocessing statistics were calculated using the entire training subset before cross-validation.

This introduced potential data leakage into the cross-validation process.

To address this, the preprocessor was moved inside the pipeline passed to `RandomizedSearchCV`.

The revised architecture was:

```python
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])
```

The hyperparameter search then operated on the complete pipeline:

```python
search_logistic_reg = RandomizedSearchCV(
    full_pipeline,
    {
        'model__C': np.logspace(-2, 1, 20)
    },
    n_iter=20,
    cv=3,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

search_logistic_reg.fit(X_train_no_proc, y_train_no_proc)
```

This ensures that imputation, scaling, and encoding are fitted independently within each cross-validation training fold.

The resulting test accuracy remained:

**93.98%**

Best hyperparameter:

```python
{'model__C': 10.0}
```

Cross-validation accuracy:

**93.37%**

Test accuracy:

**93.98%**

The relatively small difference between cross-validation and test accuracy was encouraging.

## 6. Classification Report

The standard Logistic Regression model was evaluated using `classification_report`.

Results:

| Outfit_Code | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.95 | 0.98 | 0.96 | 329 |
| 1 | 0.75 | 0.71 | 0.73 | 17 |
| 3 | 0.91 | 0.89 | 0.90 | 115 |
| 9 | 0.82 | 0.67 | 0.74 | 55 |
| 12 | 0.96 | 0.81 | 0.88 | 31 |
| 13 | 0.85 | 0.85 | 0.85 | 26 |
| 18 | 0.57 | 0.57 | 0.57 | 7 |
| 21 | 0.84 | 0.94 | 0.89 | 157 |
| 23 | 1.00 | 0.99 | 0.99 | 493 |

Overall metrics:

| Metric | Result |
|---|---:|
| Accuracy | 0.94 |
| Macro Precision | 0.85 |
| Macro Recall | 0.82 |
| Macro F1-score | 0.83 |
| Weighted F1-score | 0.94 |

### Observations

The model performed particularly well on Outfit_Code 0 and Outfit_Code 23.

However, performance was weaker for Outfit_Code 1, 9, and 18.

The support values also revealed a significant class imbalance.

For example:

- Outfit_Code 23 had 493 test examples.
- Outfit_Code 0 had 329 test examples.
- Outfit_Code 18 had only 7 test examples.

This suggests that the model has considerably more examples available to learn from for some outfit categories than others.

## 7. Experiment 5: Balanced Class Weights

To investigate the effect of class imbalance, another Logistic Regression model was trained using:

```python
class_weight="balanced"
```

This automatically adjusts class weights based on the frequency of each class in the training data.

The model configuration was:

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=5000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
```

The same regularization search was used.

### Results

| Metric | Standard Model | Balanced Model |
|---|---:|---:|
| Accuracy | 93.98% | 89.02% |
| Macro Precision | 0.85 | 0.73 |
| Macro Recall | 0.82 | 0.90 |
| Macro F1-score | 0.83 | 0.78 |
| Weighted F1-score | 0.94 | 0.90 |

The balanced model achieved substantially higher macro recall but lower overall accuracy and macro F1-score.

### Observations

The balanced model improved recall for several minority classes.

For example:

| Outfit_Code | Standard Recall | Balanced Recall |
|---|---:|---:|
| 1 | 0.71 | 1.00 |
| 9 | 0.67 | 0.87 |
| 12 | 0.81 | 0.90 |
| 18 | 0.57 | 1.00 |

However, precision decreased for some of these categories.

For Outfit_Code 18, precision decreased from 0.57 to 0.26.

This suggests that the balanced model was more likely to predict minority categories, but it also produced more false-positive predictions.

The balanced model also had lower cross-validation accuracy:

**89.28%**

Its test accuracy was:

**89.02%**

### Conclusion of Class Weighting Experiment

Class weighting improved the model's ability to identify minority categories but reduced overall predictive performance.

The standard Logistic Regression model currently provides the best balance of overall accuracy and F1-score.

Both models are worth retaining for comparison.

## 8. Overall Results

The following table summarizes the Logistic Regression experiments conducted during this session.

| Experiment | Test Accuracy |
|---|---:|
| Initial Logistic Regression | 85.77% |
| Elastic Net | 88.46% |
| L2 Regularization | 90.16% |
| One-Hot Encoding + L2 | 93.98% |
| Balanced Class Weights | 89.02% |

The best-performing configuration so far is standard Logistic Regression using:

- OneHotEncoder for categorical weather data.
- SimpleImputer for missing values.
- StandardScaler for numerical features.
- L2 regularization.
- `lbfgs` solver.
- 3-fold cross-validation.
- RandomizedSearchCV for hyperparameter tuning.
- Best regularization parameter: `C = 10.0`.

Its test accuracy is **93.98%**.

## 9. Next Steps

The next stage of the project will focus on comparing Logistic Regression against other classification algorithms.

Potential algorithms to investigate include:

1. Random Forest Classifier.
2. Decision Tree Classifier.
3. Support Vector Machine (SVM).
4. K-Nearest Neighbors (KNN).

The same train/test split and preprocessing principles should be maintained to make comparisons meaningful.

Each model should be evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix

The standard Logistic Regression model will serve as the baseline for these comparisons.

## 10. Final Summary

This session significantly improved the Logistic Regression model, increasing test accuracy from 85.77% to 93.98%.

The most notable improvement came from switching categorical weather preprocessing from ordinal encoding to one-hot encoding.

Another important improvement was moving preprocessing inside the cross-validation pipeline to prevent validation-fold leakage.

Finally, class weighting was investigated to address class imbalance. Although it improved minority-class recall, it reduced overall accuracy and macro F1-score.

The current best model is standard Logistic Regression with an accuracy of 93.98%.

The next objective is to investigate whether nonlinear classification algorithms, particularly Random Forest, can improve upon this baseline.
