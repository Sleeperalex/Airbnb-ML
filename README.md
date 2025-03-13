# Airbnb Price Prediction Project

## Overview
The **Airbnb Price Prediction Project** aims to predict the log-transformed price (`log_price`) of Airbnb listings based on various features such as location, property type, amenities, and more. The project involves data cleaning, exploratory data analysis (EDA), feature engineering, and the implementation of several machine learning models to predict Airbnb prices accurately.

---

## Project Steps

### 1. Data Loading and Initial Exploration
- **Objective**: Load the dataset and perform initial exploration to understand the structure and content of the data.
- **Key Steps**:
  - Load the dataset using `pandas`.
  - Display the first few rows of the dataset to understand the features.
  - Check for missing values and data types.
- **Observations**:
  - The dataset contains both numerical and categorical features.
  - Some columns contain text data that needs to be converted into numerical values for modeling.

### 2. Data Cleaning and Preprocessing
- **Objective**: Clean and preprocess the dataset to handle missing values, encode categorical variables, and standardize numerical features.
- **Key Steps**:
  - **Class `Netoyage`**: A custom class is created to handle data cleaning and preprocessing.
    - **Methods**:
      - `convert_text`: Converts categorical text columns into numerical values.
      - `convert_rate`: Converts the `host_response_rate` column into a numerical percentage.
      - `convert_bool`: Converts boolean text columns into numerical values (1 for `True`, 0 for `False`).
      - `convert_date`: Converts date columns into the number of days since the current date.
      - `replace_caracter`: Replaces special characters in column names with underscores.
      - `drop_unnecessary_columns`: Drops columns that are not useful for modeling (e.g., `id`, `description`, `name`, `zipcode`).
      - `handle_missing_values`: Fills missing values with 0.
      - `create_amenities_columns`: Creates binary columns for each unique amenity in the `amenities` column.
  - **Result**: The dataset is cleaned and transformed into a format suitable for machine learning.

### 3. Exploratory Data Analysis (EDA)
- **Objective**: Perform EDA to understand the relationships between features and the target variable (`log_price`).
- **Key Steps**:
  - **Correlation Analysis**: Calculate the correlation matrix and visualize it using a heatmap.
  - **Distribution Plots**: Plot the distribution of numerical features to understand their spread and skewness.
  - **Geographical Visualization**: Create a map using `folium` to visualize Airbnb listings based on their location and price.
- **Observations**:
  - The `log_price` distribution is bimodal, indicating two main price groups.
  - Features like `room_type`, `accommodates`, and `bathrooms` show significant correlations with `log_price`.
  - The map visualization reveals that higher-priced listings are concentrated in specific areas.

### 4. Model Implementation and Evaluation
- **Objective**: Implement and evaluate several machine learning models to predict `log_price`.
- **Key Steps**:
  - **Data Splitting**: Split the dataset into training and testing sets (80% training, 20% testing).
  - **Model Training**: Train the following models:
    - **Linear Regression**: A simple linear model.
    - **Random Forest Regressor**: An ensemble model using decision trees.
    - **Gradient Boosting Regressor**: A boosting model that builds trees sequentially.
    - **XGBoost**: An optimized gradient boosting model.
  - **Model Evaluation**: Evaluate each model using the R² score on both the training and testing sets.
- **Results**:
  - **XGBoost** performed the best, with an R² score of 0.90 on the training set and 0.71 on the testing set.
  - **Random Forest** and **Gradient Boosting** also performed well but were slightly less accurate than XGBoost.
  - **Linear Regression** had the lowest performance, indicating that the relationship between features and `log_price` is non-linear.

### 5. Prediction on Test Data
- **Objective**: Use the best-performing model (XGBoost) to predict `log_price` on a separate test dataset.
- **Key Steps**:
  - Load and preprocess the test dataset using the same `Netoyage` class.
  - Align the test dataset columns with the training dataset.
  - Make predictions using the trained XGBoost model.
  - Save the predictions to a CSV file (`predictions.csv`).
- **Result**: The predictions are saved and can be used for further analysis or submission.

---

## How to Run the Code

1. **Install Dependencies**:
   - Ensure you have Python 3.x installed.
   - Install the required libraries using:
     ```bash
     pip install pandas numpy scikit-learn xgboost folium matplotlib seaborn
     ```

2. **Download the Dataset**:
   - Place the `airbnb_train.csv` and `airbnb_test.csv` files in the project directory.

3. **Run the Script**:
   - Execute the provided Python script to preprocess the data, train the models, and evaluate their performance.

4. **View Results**:
   - The script will generate predictions and save them to `predictions.csv`.
   - You can also visualize the results using the generated plots and maps.

---

## Results and Interpretation

### Model Performance
- **XGBoost** achieved the highest R² score on both the training and testing sets, indicating that it is the best model for this dataset.
- The R² score on the testing set (0.71) suggests that the model explains 71% of the variance in `log_price`, which is a strong performance for a real-world dataset.

### Feature Importance
- Features like `room_type`, `accommodates`, and `bathrooms` were found to be highly correlated with `log_price`.
- The `amenities` feature, after being transformed into binary columns, also contributed significantly to the model's performance.

### Limitations
- The model may overfit the training data, as indicated by the higher R² score on the training set compared to the testing set.
- Some features, such as `host_response_rate`, had missing values that were filled with 0, which may not accurately reflect the true distribution of the data.
