Deployed on Render
🔗 <a href="https://calorie-expenditure-predictor-app.onrender.com/" target="_blank" rel="noopener noreferrer">
Calorie Expenditure Predictor App
</a>

# Calorie Expenditure Prediction
This project predicts the number of calories burned during a workout session using physiological and activity-related features such as Age, Height, Weight, Duration, Heart Rate, and Body Temperature.

By leveraging machine learning models, the system identifies which factors most strongly influence calorie expenditure and selects the best-performing predictive algorithm for fitness planning and performance tracking.

The dataset, sourced from Kaggle’s [Calories Burnt Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction/data) collection, provides the foundation for training a predictive system that supports personalised fitness insights and healthier lifestyle decisions.

[Watch the Demo](Streamlit%20App%20Demo.mp4)

# Dataset Description
 - Shape: 15000 rows, 7 columns
 - Features:
   - User_ID: unique identifier for each individual
   - Gender: Male/Female
   - Age: individual's age (20 - 79 years)
   - Height: cm
   - Weight: kg
   - Duration: minutes
   - Heart_Rate: average heart rate during workout
   - Body_Temp: body temperature during workout
  - Target
    - Calories

# Exploratory Data Analysis (EDA) Insights
- Average participant: 174 cm, 75 kg, 96 bpm, 15.5 min workout, ~90 kcal burned.
- Strong correlations: Duration, Heart Rate, Body Temperature with Calories.
- Height & Weight correlation: 0.958 → engineered BMI feature.
- Gender differences: Males are slightly taller/heavier.
- Outliers in height/weight detected.

# Feature Engineering and Selection
- Created `bmi = (Weight / Height (m) ** 2)`
- Selected Features:
   -  X = Age, Gender, Duration, Heart_Rate, Body_Temp, bmi
   -  y = Calories
- Drop identifier `User_ID`
- OneHotEncoding `Gender` as numeric
- Standscale the numeric features; Age, Duration, Heart_Rate, Body_Temp, bmi

# Modelling - Training, Evaluation and Selection

| Model                  | Metrics (MAE - RMSE - R²) | Remark                                     |
| ---------------------- | ------------------------- | ------------------------------------------ |
| **Ridge Regression**   | 8.460 - 11.597 - 0.966    | Model with regularization                             |
| **Linear Regression**  | 8.460 - 11.597 - 0.966    | Baseline model |
| **Decision Tree**      | 3.964 - 6.003 - 0.991     | Improved accuracy with slight overfitting  |
| **K Nearest Neighbor** | 4.237 - 5.842 - 0.991     | Improved accuracy                          |
| **Gradient Boosting**  | 3.228 - 4.474 - 0.995     | Best generalizing model                    |
| **Random Forest**      | 2.562 - 3.856 - 0.996     | Best predictive accuracy                   |

**Selected Models:** Random Forest Regressor and Gradient Boosting

# Hyperparameter Tuning:
To further improve model performance, I applied GridSearchCV to the top two models: Random Forest and Gradient Boosting.
| Model                 | Best Parameters                                                                                                                                         |        MAE |      RMSE |        R² |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------: | --------: | --------: |
| **Random Forest**     | `{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`                                         |     2.835 |     4.413 |     0.995 |
| **Gradient Boosting** | `{'learning_rate': 0.05, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.8}` | **2.491** | **3.500** | **0.997** |

Gradient Boosting achieved lower MAE and RMSE values with better generalisation, demonstrating superior performance and robustness.
**Final Model Selection:** Tuned Gradient Boosting

# Model Interpretation
To interpret how features influenced calorie predictions, two complementary explainability methods were applied — Permutation Importance and SHAP (SHapley Additive Explanations).
These methods were used on the final Gradient Boosting model, the best-performing estimator after hyperparameter tuning.

## Permutation Importance
| Feature        | Importance |
| -------------- | ---------: |
| **Duration**   | **1.0731** |
| **Heart_Rate** | **0.1691** |
| **Age**        |     0.0563 |
| **Gender**     |     0.0113 |
| **BMI**        |     0.0016 |
| **Body_Temp**  |    0.00005 |
    
**Interpretation:** Workout Duration and Heart Rate dominate calorie prediction. Physiological attributes (BMI, Age, Gender) contribute marginally.

# Future Improvements
- Deploy advanced models (Polynomial Regression, Neural Networks).
- Improve UI with interactive charts & personalised recommendations.
