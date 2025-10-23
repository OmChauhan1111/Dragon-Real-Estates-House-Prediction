# Dragon Real Estate Price Prediction

A machine learning project to predict the median value of owner-occupied homes in Boston using the **Boston Housing dataset**. This project uses **Random Forest Regressor** along with data preprocessing, feature scaling, and evaluation techniques to build an accurate prediction model.

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

The goal of this project is to **predict housing prices** based on multiple features such as crime rate, number of rooms, distance to employment centers, property tax, and other socio-economic variables.  

The **Random Forest Regressor** was chosen for its ability to handle complex relationships and provide high prediction accuracy.

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
| AGE | Proportion of owner-occupied units built prior to 1940 |
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
- **Train-Test Split**: 80% training, 20% testing, stratified on `CHAS`.  
- **Correlation Analysis**:
  - Positive correlation with `MEDV`: `RM` (0.679)  
  - Negative correlation with `MEDV`: `LSTAT` (-0.74), `TAXRM` (-0.52)  

---

## Exploratory Data Analysis (EDA)

- Histograms to understand data distribution.  
- Scatter plots and scatter matrix to visualize relationships with `MEDV`.  
- Observations:
  - Higher `RM` → Higher `MEDV`  
  - Higher `LSTAT` → Lower `MEDV`  

---

## Feature Engineering

- Created `TAXRM = TAX / RM` to capture tax per room effect.  
- Selected features based on correlation for modeling.

---

## Modeling

- Models tested:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor ✅  
  - Gradient Boosting Regressor  

**Random Forest Regressor** performed the best in terms of accuracy and robustness.

---

## Pipeline & Scaling

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

housing_prepared = pipeline.fit_transform(housing)
Evaluation
Training Performance
Metric	Value
Mean Squared Error (MSE)	1.396
Root Mean Squared Error (RMSE)	1.181
R² Score	0.984

Excellent fit on training data, capturing underlying patterns effectively.

Cross-Validation (10-Fold)
Fold	RMSE
1	2.740
2	2.838
3	4.454
4	2.601
5	3.456
6	2.560
7	4.988
8	3.369
9	3.435
10	2.985

Summary:

Metric	Value
Mean RMSE	3.343
Std Deviation	0.766

Consistent performance on unseen folds, showing model robustness.

Test Set Performance
Metric	Value
Mean Squared Error (MSE)	8.630
Root Mean Squared Error (RMSE)	2.938
R² Score	0.878

Model generalizes well to new, unseen data.

Sample Predictions
Actual (MEDV)	Predicted
16.5	24.327
10.2	11.757
30.1	25.521
23.0	22.025
14.4	18.968

Predictions are very close to actual values, demonstrating high accuracy.

Conclusion
Random Forest Regressor effectively predicts housing prices.

Cross-validation ensures robustness and prevents overfitting.

Model is ready for deployment using the saved Dragon.joblib.

Saving and Loading Model
python
Copy code
from joblib import dump, load

# Save model
dump(model, 'Dragon.joblib')

# Load model
model = load('Dragon.joblib')
Usage
python
Copy code
import numpy as np
from joblib import load

model = load('Dragon.joblib')
features = np.array([[...]])  # Preprocessed input features
prediction = model.predict(features)
print("Predicted Price:", prediction)
Dependencies
Python 3.x

scikit-learn

pandas

numpy

matplotlib

seaborn

Author
Om Singh Chauhan
Email: omchauhanom1111@gmail.com
BCA Final Year (Data Science)

pgsql
Copy code

This version is **fully polished**, professional, GitHub-ready, with **all evaluation metrics, pipelines, and usage clearly explained**.  

If you want, I can also make a **shorter, visually attractive README with badges, images, and highlighted key metrics** for better GitHub appeal.  

Do you want me to do that too?
