# Calorie Expenditure Prediction
This project aims to predict the number of calories burned during a workout session based on physiological and activity-related features such as: Age, Height, Weight, Duration, Heart Rate and Body Temperature. 
<br>It uses a machine learning model to understand which factors most influence calorie expenditure and determine the best performing  predictive model for fitness planning and performance tracking.</br>
<br>The dataset, sourced from Kaggle's [Calories Burnt Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction/data) collection, provides the foundation for training a predictive system that can support personalised fitness insights and healthier lifestyle decisions.</br>

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

# Insights from Exploratory Data Analysis
- The dataset contains 15,000 workout sessions from adults aged 20 to 79, with participants averaging 174 cm in height, 75 kg in weight, and a heart rate of about 96 bpm; workouts last around 15.5 minutes on average, resulting in approximately 90 calories burned per session, with some variability and a few extreme height and weight values suggesting potential outliers.
- Heart rate, duration and body temperature show strong positive correlation with calories burned.
- Weight and Height show strong correlation (0.958)
- Males are slightly taller and heavier than females.

# Feature Engineering
Due to strong multicollinearity between Height and Weight (0.958), I created the composite feature `bmi = (Weight / Height (m) ** 2)`

# Feature Selection & Pre-modelling Data Prep
  - Selected Features:
      -  X = Age, Gender, Duration, Heart_Rate, Body_Temp, bmi
      -  y = Calories
  - Drop identifier `User_ID`
  - OneHotEncoding `Gender` as numeric
  - Standscale the numeric features; Age, Duration, Heart_Rate, Body_Temp, bmi

# Modelling - Training, Evaluation and Selection
To determine the best predictive algorithm, multiple regression models were trained:

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

# Interpretation
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
    
**Interpretation:** The permutation importance analysis from the tuned Gradient Boosting model reveals that Workout Duration is by far the most influential factor determining calorie expenditure, followed by Heart Rate. Together, these two variables account for nearly all the model’s explanatory power.
Physiological attributes such as BMI, Age, and Gender contribute marginally, indicating that workout behavior and exertion level dominate over body characteristics in calorie prediction.
This is also consistent with the result from SHAP analysis.

# Future Improvements
- Deploy web app for real-time calorie prediction using user input
- try out polynomial regression or Neural Networks for complex interactions.
