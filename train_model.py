import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib


def train_and_save_model():
    # Load dataset
    df = pd.read_csv("df_clean.csv")

    # Convert datetime columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

    # Compute target: delivery time
    df['delivery_time'] = (
        (df['actual_delivery_time'] - df['created_at'])
        .dt.total_seconds() / 60
    ).round(2)

    # Remove outliers using IQR
    Q1 = df['delivery_time'].quantile(0.25)
    Q3 = df['delivery_time'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df = df[df['delivery_time'] < upper_bound]

    # Feature engineering
    df['hour'] = df['created_at'].dt.hour
    df['days_of_week'] = df['created_at'].dt.dayofweek
    df['is_weekend'] = df['days_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Features
    features = [
        'hour', 'days_of_week', 'is_weekend', 'total_items',
        'num_distinct_items', 'subtotal', 'min_item_price',
        'max_item_price', 'total_onshift_partners',
        'total_busy_partners', 'total_outstanding_orders'
    ]

    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split data
    X = df[features]
    y = df['delivery_time']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBRegressor
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"Root Mean Squared Error: {rmse:.2f} minutes")

    # Save model + scaler + features
    joblib.dump({
        "model": xgb,
        "scaler": scaler,
        "features": features
    }, "xgb_regressor_model.pkl")

    print("✅ XGBRegressor model saved to xgb_regressor_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
