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


**Evaluation
**
**Training Set:**
MSE: 1.396
RMSE: 1.181
R² Score: 0.984

**Cross Validation (10-fold):**
Mean RMSE: 3.34
Std Dev: 0.77

**Test Set:**
MSE: 8.63
RMSE: 2.94
R² Score: 0.878

Saving and Loading Model
from joblib import dump, load

** Save model**
dump(model, 'Dragon.joblib')

# Load model for prediction
model = load('Dragon.joblib')
features = np.array([[...]])
model.predict(features)


**Author**

Om Singh Chauhan
BCA 2nd Year (Data Science)
Email: omchauhanom1111@gmail.com
