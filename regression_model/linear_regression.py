import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Load the data
df = pd.read_csv('data/temp_features.csv')

# Assign X
x = df[['depth',
        'height',
        'Anemometer;wind_speed;Avg (m/s)',
        'Wind Vane TMR;wind_direction;Avg (°)',
        'Hygro/Thermo;humidity;Avg (%)',
        'Hygro/Thermo;temperature;Avg (°C)',
        'Barometer;air_pressure;Avg (hPa)',
        'DNI (Direct Normal Irradiance) Pyrheliometer;solar_DNI;Avg (W/m²)',
        'north',
        'west',
        'east',
        # 'south',
        'hour',
        'day'
        ]].copy()

# Assign Y
y = df['Temperature'].copy()

# Train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# Pipline
lr_pipeline = make_pipeline(StandardScaler(), LinearRegression())
lr_pipeline.fit(x_train, y_train)

# Extract coefficients
model_lr = lr_pipeline.named_steps['linearregression']
coefficients_lr = model_lr.coef_

# Predict
y_pred_lr = lr_pipeline.predict(x_test)

# Evaluate
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = math.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Predict w/ the Dummy model
dummy_pipeline = make_pipeline(StandardScaler(), DummyRegressor(strategy='mean'))
dummy_pipeline.fit(x_train, y_train)
y_pred_dummy = dummy_pipeline.predict(x_test)

# Evaluate the Dummy model
mse_dummy = mean_squared_error(y_test, y_pred_dummy)
rmse_dummy = math.sqrt(mse_dummy)
mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
r2_dummy = r2_score(y_test, y_pred_dummy)


# Print stats
results = pd.DataFrame({
    'Feature': x.columns,
    'Coefficients': coefficients_lr
})
performance_diff = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R² score'],
    'Dummy Regressor': [mse_dummy, rmse_dummy, mae_dummy, r2_dummy],
    'Linear Regression': [mse_lr, rmse_lr, mae_lr, r2_lr],
    'Difference': [mse_lr - mse_dummy, rmse_lr - rmse_dummy, mae_lr - mae_dummy, 'n/a']
})

print("Feature Coefficients:")
print(results)
print("\nPerformance Comparison:")
print(performance_diff)

results.to_csv("feature_coef.csv")
performance_diff.to_csv("performance.csv")
