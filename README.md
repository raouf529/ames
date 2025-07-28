# Ames Housing Price Prediction

This project builds a machine learning model to predict house prices using the Ames Housing Dataset from Kaggle.

## 📊 Dataset

- **Source**: [Ames Housing Kaggle Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- 80+ features including:
  - Neighborhood, square footage, number of bedrooms/bathrooms, year built, garage, etc.

## ⚙️ Tools & Libraries

- Python 3
- Scikit-Learn
- Pandas, NumPy
- Matplotlib / Seaborn
- Joblib

## 🧠 Modeling Process

1. **Preprocessing Pipeline**:
   - Imputation of missing values
   - Feature scaling
   - One-hot encoding for categorical features

2. **Model Selection**:
   - RandomForestRegressor
   - Hyperparameter tuning using `RandomizedSearchCV`

3. **Evaluation**:
   - Cross-validation RMSE (mean ± std): `25329.46 ± 4318.89`
   - Final test RMSE: `21740.69`
   - 95% Confidence Interval for RMSE: `(97720.69, 110215.70)` (for squared errors, not directly RMSE)

## 📈 Example Prediction

| Feature             | Value     |
|---------------------|-----------|
| Overall Quality      | 8         |
| Living Area (sq ft)  | 2000      |
| Year Built           | 2005      |
| Garage               | 2 cars    |
| Bedrooms             | 3         |
| Full Baths           | 2         |

**Predicted Sale Price**: `$235,683.60`

## 💾 Saved Models

- `random_forest_final_model.pkl`: Trained model
- `preprocessing_pipeline.pkl`: Full preprocessing pipeline (for use on new data)

## 📂 How to Use

```python
import joblib
model = joblib.load("random_forest_final_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

X_new = ... # new data as DataFrame
X_prepared = pipeline.transform(X_new)
predictions = model.predict(X_prepared)

