# Ames Housing Price Prediction

This project builds a machine learning model to predict house prices using the Ames Housing Dataset from Kaggle.

## 📊 Dataset

- **Source**: [Ames Housing Kaggle Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- 80+ features including:
  - Neighborhood, square footage, number of bedrooms/bathrooms, year built, garage, etc.
- **Time Period**: 2006–2010 (historical prices, not current market)

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
   - add some features(fail and drop the idea)

2. **Model Selection**:
    -linear models: linear regression, Lasso, Ridge
   - RandomForestRegressor
   - Hyperparameter tuning using `RandomizedSearchCV`

4. **Evaluation**:  
    - **Test RMSE**: $21,618 (on held-out data)  
    - **95% RMSE Confidence Interval**: $18,647 – $24,865  
      - *Interpretation*: For 95% of similar homes, prediction errors fall in this range.  

## 📈 Example Prediction

| Feature             | Value     |
|---------------------|-----------|
| Overall Quality      | 8         |
| Living Area (sq ft)  | 2000      |
| Year Built           | 2000      |
| Garage               | 2 cars    |
| Bedrooms             | 3         |
| Full Baths           | 2         |

**Predicted Sale Price**: `$241,737.67`

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

