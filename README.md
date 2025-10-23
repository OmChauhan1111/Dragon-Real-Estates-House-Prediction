# Dragon Real Estate Price Prediction

A machine learning project to predict the median value of owner-occupied homes in Boston using the Boston Housing dataset. The project uses **Random Forest Regressor** along with data preprocessing, feature scaling, and evaluation techniques to build an accurate prediction model.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Preprocessing](#data-preprocessing)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Feature Engineering](#feature-engineering)  
- [Modeling](#modeling)  
- [Pipeline & Scaling](#pipeline--scaling)  
- [Evaluation](#evaluation)  
- [Saving and Loading Model](#saving-and-loading-model)  
- [Usage](#usage)  
- [Dependencies](#dependencies)  
- [Author](#author)  

---

## Project Overview

This project predicts housing prices based on multiple features such as crime rate, number of rooms, distance to employment centers, property tax, and other socio-economic variables.  

The **Random Forest Regressor** was selected for its ability to handle complex data patterns and provide high prediction accuracy.

---

## Dataset

The **Boston Housing dataset** contains 506 rows and 14 columns:

| Column | Description |
|--------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river, 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AG | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centers |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
| LSTAT | % lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000's |

---

## Data Preprocessing

- **Missing Values**: Filled using `SimpleImputer(strategy='median')`.
- **Train-Test Split**: 80% training, 20% testing with stratified sampling on `CHAS`.
- **Correlation Analysis**:
  - Strong positive correlation with `MEDV`: `RM` (0.679)
  - Strong negative correlation with `MEDV`: `LSTAT` (-0.74), `TAXRM` (-0.52)

---

## Exploratory Data Analysis (EDA)

- Histograms plotted to understand distribution.
- Scatter plots and `scatter_matrix` visualized relationships with `MEDV`.
- Observations:
  - More rooms (`RM`) → higher `MEDV`
  - Higher % lower status (`LSTAT`) → lower `MEDV`

---

## Feature Engineering

- Created a new feature `TAXRM = TAX / RM` to capture tax per room effect.
- Selected features with strong correlation for modeling.

---

## Modeling

- Models tested:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor ✅
  - Gradient Boosting Regressor
- **Random Forest Regressor** selected due to best performance.

---

## Pipeline & Scaling

```python
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

housing_num_tr = my_pipeline.fit_transform(housing)


# Dragon Real Estate Price Prediction

A machine learning project to predict the median value of owner-occupied homes in Boston using **Random Forest Regressor**.

---

## Model Evaluation

### Training Performance

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 1.396 |
| Root Mean Squared Error (RMSE) | 1.181 |
| R² Score | 0.984 |

> The model fits the training data very well, capturing trends with high accuracy.

---

### Cross-Validation (10-Fold)

| Fold | RMSE |
|------|------|
| 1 | 2.740 |
| 2 | 2.838 |
| 3 | 4.454 |
| 4 | 2.601 |
| 5 | 3.456 |
| 6 | 2.560 |
| 7 | 4.988 |
| 8 | 3.369 |
| 9 | 3.435 |
| 10 | 2.985 |

**Summary:**

| Metric | Value |
|--------|-------|
| Mean RMSE | 3.343 |
| Standard Deviation | 0.766 |

> Cross-validation shows the model performs consistently on unseen folds, with slight variations due to data distribution.

---

### Test Set Performance

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 8.630 |
| Root Mean Squared Error (RMSE) | 2.938 |
| R² Score | 0.878 |

> The test set performance confirms that the model generalizes well to new data.

---

### Sample Predictions

| Actual (MEDV) | Predicted |
|---------------|-----------|
| 16.5 | 24.327 |
| 10.2 | 11.757 |
| 30.1 | 25.521 |
| 23.0 | 22.025 |
| 14.4 | 18.968 |

> The model predictions are very close to actual values, demonstrating strong predictive capability.

---

### Conclusion

- The **Random Forest Regressor** accurately predicts housing prices.
- Cross-validation ensures robustness and prevents overfitting.
- The model can be deployed for real-time predictions using the saved `Dragon.joblib`.

---

## Usage

```python
from joblib import load
import numpy as np

model = load('Dragon.joblib')
features = np.array([[...]])  # Preprocessed features
prediction = model.predict(features)
print("Predicted Price:", prediction)


Author

Om Singh Chauhan
Email: omchauhanom1111@gmail.com

BCA Final Year (Data Science)
