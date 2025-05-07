# Smart Factory Energy Prediction Challenge - Final Report

## 1. Introduction & Problem Overview

SmartManufacture Inc. aims to predict equipment energy consumption in a client's manufacturing facility using sensor data. This report details the development of a machine learning model to achieve this, focusing on data analysis, model building, evaluation, and providing actionable insights for energy optimization. The primary goal is to help facility managers reduce energy costs and improve operational efficiency.

## 2. Approach

The project followed a structured data science workflow:

1.  **Data Loading and Initial Inspection**: The dataset (`data.csv`) was loaded. Initial checks included data shape, types, missing values, and basic statistics. Object-type columns intended for numeric data were identified.
2.  **Exploratory Data Analysis (EDA)**:
    *   The `timestamp` column was converted to datetime objects.
    *   Relevant object-type columns (e.g., `equipment_energy_consumption`, sensor readings) were converted to numeric types, with errors coerced to NaN.
    *   The distribution of the target variable (`equipment_energy_consumption`) was analyzed, revealing a moderate right-skewness (skewness ~0.62).
    *   Correlation analysis (heatmap) was performed to understand relationships between features.
    *   Initial outlier assessment using descriptive statistics and boxplots highlighted implausible values in some sensor readings (e.g., negative humidity).
3.  **Data Preprocessing**:
    *   Rows with missing target values (`equipment_energy_consumption`) were dropped.
    *   Remaining missing values in numerical features were imputed using median imputation, chosen for its robustness to outliers.
    *   Outliers were addressed:
        *   Humidity features were clipped to the realistic 0-100% range.
        *   Other sensor readings (temperatures, lighting energy, wind speed, visibility) were capped using 1st and 99th percentiles to mitigate extreme values.
4.  **Feature Engineering**:
    *   Time-based features were extracted from the `timestamp`: `hour`, `day_of_week` (0-6), `day_of_month`, `month`, `year`.
    *   A binary `is_weekend` feature was created.
    *   Cyclical encoding (sine/cosine transformations) was applied to `hour`, `day_of_week`, and `month` to better capture their cyclical nature for machine learning models.
5.  **Feature Selection**:
    *   The original `timestamp` column and non-cyclical raw time features (`hour`, `day_of_week`, `month`, `day_of_month`) were dropped in favor of the engineered cyclical features and `year`/`is_weekend`.
    *   `random_variable1` and `random_variable2` were analyzed:
        *   Correlation with the target variable was found to be negligible (close to 0).
        *   Scatter plots showed no discernible pattern against the target.
        *   Feature importance from a LightGBM model (a powerful tree-based model) consistently ranked these variables very low.
        *   Based on this evidence, `random_variable1` and `random_variable2` were excluded from the final feature set as they provided no predictive value.
6.  **Model Development and Training**:
    *   The preprocessed data was split into training (80%) and testing (20%) sets.
    *   Numerical features were scaled using `StandardScaler`.
    *   Several regression models were trained and evaluated: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
    *   LightGBM, being one of the top-performing baseline models, was selected for hyperparameter tuning using `RandomizedSearchCV` (with 25 iterations and 3-fold cross-validation) to optimize its performance based on the R² scoring metric.
7.  **Model Evaluation**:
    *   All models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) on the test set.

## 3. Key Insights from the Data

*   **Target Variable**: `equipment_energy_consumption` is moderately right-skewed. Most consumption values are relatively low, but there's a tail of higher consumption instances.
*   **Influential Factors**:
    *   Humidity levels across various zones (e.g., `zone2_humidity`, `zone9_humidity`, `zone5_humidity`, `outdoor_humidity`) and `dew_point` consistently showed high importance in the LightGBM model.
    *   Temperatures in different zones (e.g., `zone1_temperature`, `zone8_temperature`) were also significant predictors.
    *   Other features like `wind_speed` and cyclical time representations also contributed.
*   **Data Quality**: The raw dataset required significant cleaning. Incorrect data type detection for numeric columns, widespread missing values, and physically implausible outliers (like negative humidity or extreme temperatures) were key challenges addressed during preprocessing.
*   **Random Variables**: `random_variable1` and `random_variable2` were determined to be non-informative for predicting energy consumption due to near-zero correlation with the target and negligible feature importance.

## 4. Model Performance Evaluation

The following table summarizes the performance of the baseline models and the tuned LightGBM model on the test set:

| Model                 | RMSE      | MAE     | R²    |
| :-------------------- | :-------- | :------ | :------ |
| **LightGBM (Tuned)**  | **155.105** | **60.816** | **0.104** |
| LightGBM (Baseline)   | 155.825   | 64.912  | 0.096   |
| Gradient Boosting     | 157.217   | 65.017  | 0.079   |
| Random Forest         | 159.785   | 67.135  | 0.049   |
| Ridge Regression      | 160.646   | 70.265  | 0.039   |
| Linear Regression     | 160.646   | 70.266  | 0.039   |
| XGBoost (Baseline)    | 166.405   | 78.665  | -0.031  |

The tuned LightGBM model yielded the best performance, although the R² score of 0.104 indicates that the model explains only about 10.4% of the variance in equipment energy consumption. This suggests that while the model has learned some patterns, a large portion of the variability is not captured by the current feature set.

## 5. Recommendations for Reducing Equipment Energy Consumption

Given the model's performance and feature importances, the following recommendations are made:

1.  **Prioritize Humidity and Temperature Management**:
    *   The high importance of humidity (various zones, outdoor, dew point) and temperatures suggests that optimizing HVAC systems and environmental controls in these specific zones can lead to energy savings.
    *   Implement precise monitoring and control, potentially using the model's predictions to anticipate loads.
2.  **Address Dew Point Impact**:
    *   Since `dew_point` is a key factor, investigate processes or areas sensitive to absolute moisture levels. Targeted dehumidification or improved ventilation strategies might be beneficial.
3.  **Weather-Adaptive Controls**:
    *   The influence of `outdoor_humidity`, `outdoor_temperature`, and `wind_speed` highlights the need for systems that can adapt to external weather conditions. This could involve predictive controls based on weather forecasts.
4.  **Investigate Zone-Specific Anomalies**:
    *   Focus on zones whose sensor readings (e.g., `zone2_humidity`, `zone9_humidity`, `zone1_temperature`) are highly ranked by the model. Unusually high or low readings in these critical zones, when correlated with energy spikes, could point to localized inefficiencies or faulty equipment.
5.  **Improve Data Collection for Better Prediction**:
    *   **Crucially, the low R² score indicates that essential predictive information is likely missing.** To significantly improve the model's predictive power, consider incorporating:
        *   **Operational Data**: Machine runtimes, production schedules, equipment load levels, specific processes active in different zones.
        *   **Equipment-Specific Data**: Age, efficiency ratings, and maintenance logs for major energy-consuming equipment.
        *   **Occupancy Data**: Real-time or scheduled occupancy levels in different factory zones.
6.  **Use Model for Anomaly Detection (with caution)**:
    *   Even with a low R², the current model can identify deviations from expected energy consumption based on the available sensor data. Large, unexpected spikes could trigger alerts for investigation, potentially catching equipment malfunctions early.
7.  **Continuous Monitoring and Model Refinement**:
    *   Periodically retrain the model with new data. As more comprehensive data (as suggested above) becomes available, incorporate it to build more accurate and insightful models.

## 6. Conclusion

The developed LightGBM model provides a foundational predictive capability for equipment energy consumption. While its current R² score is modest (0.104), the analysis has successfully identified key environmental drivers, particularly humidity and temperature across various zones, and the non-utility of the provided random variables. The most significant path to improving predictive accuracy and deriving more impactful energy-saving strategies lies in enriching the dataset with operational and equipment-specific features. The current model can still serve as a useful tool for baseline monitoring and identifying areas for initial optimization in HVAC and environmental controls.