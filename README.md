# calorie-expenditure-prediction
This project aims to build a machine learning model to predict an individual’s calorie expenditure based on physiological and activity-related features. 
The dataset, sourced from Kaggle's [Calories Burnt Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction/data) collection, provides the foundation for training a predictive system that can support personalised fitness insights and healthier lifestyle decisions.

# Data Cleaning & Preprocessing
To ensure data quality for machine learning modelling, the following steps were undertaken;
- checked for missing/null values:
  ``` python
  df.isnull().sum()
  ```
- checked and removed duplicates:
  ```python
  df = df.drop_duplicates(keep = 'first')
  ```
- categorical variable 'Gender' was transformed into a numerical column 'Male' where male = 1 and female = 0 using
   ```python
   encoded_gender = pd.get_dummies(df['Gender'], drop_first=True, dtype = 'int')
   df = pd.concat([df, encoded_gender], axis = 1).drop(['Gender', 'Age_Group'], axis = 1)
   ```
- train test split
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101
  ```
- standard scaling
  ```python
   from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  ```

# Insights from Exploratory Data Analysis
- There are 7553 female to 7447 male in the dataset
- Duration is the primary driver of calorie expenditure across both genders.
- There’s no significant difference in how long males and females exercise, but the calories burned differ slightly, possibly due to intensity or physical differences.
- Older individuals tend to burn slightly more calories during exercise, with males generally burning

# Feature Selection
- X = Age, Height, Weight, Duration, Heart_Rate, Body_Temp
- y = Calories

# Model Selection, Training 
Ridge regression was selected as the primary predictive model for this project due to its ability to handle multicollinearity among input features (Weight and Height (0.96) and prevent overfitting through L2 regularisation. 
Given the continuous nature of the target variable (calories burned), Ridge regression offers a balanced approach that enhances model generalisation while maintaining interpretability. 
```python
  from sklearn.linear_model import Ridge
  base_model = Ridge()
  base_model.fit(X_train, y_train)
```

# Model Evaluation
The Ridge regression model demonstrated strong predictive accuracy on the test dataset. The **Root Mean Squared Error (RMSE) was approximately 11.35**, indicating a low average prediction error in calorie estimates. Additionally, the model achieved a high **R² Score of 0.967**, suggesting that the input features explain over 96% of the variance in calorie expenditure. These metrics confirm that Ridge regression is well-suited for this task, offering both accuracy and generalizability.

# Hyperparameter Tuning with GridSearchCV
Ridge regression parameters were optimised using GridSearchCV with 5-fold cross-validation. 
```python
param_grid = {
'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
'solver': ['auto', 'cholesky', 'lsqr', 'sag', 'saga']
}

grid_model = GridSearchCV(
estimator = base_model,
param_grid = param_grid,
cv = 5,
scoring = 'neg_root_mean_squared_error',
n_jobs = -1
)
```
The best configuration: alpha = 0.1 and solver = 'sag'—produced a cross-validated RMSE of approximately 11.30, enhancing the model’s predictive accuracy and generalisation.

# Feature Importance
The model identified *Duration* and *Heart Rate* as the strongest predictors of calorie expenditure. 
Moderate contributions came from *Age* and *Weight*, while features like *Gender*, *Height*, and *Body Temperature* had minimal or negative influence.

| Feature       | Coefficient |
|---------------|-------------|
| Duration      | 55.40       |
| Heart_Rate    | 19.10       |
| Age           | 8.50        |
| Weight        | 4.36        |
| Male          | -0.58       |
| Height        | -2.56       |
| Body_Temp     | -13.25      |

# Conclusion
This project successfully demonstrated that Ridge regression, especially when fine-tuned using GridSearchCV, can accurately predict calorie expenditure from physiological and activity-related data. 
Key findings emphasised the dominant role of Duration and Heart Rate in estimating energy burn, offering practical insight for developing personalised health and fitness applications
