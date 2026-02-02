import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class RegressionMastery:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.best_model = None
        self.best_scaler = None
        
    def check_vif(self, X):
        """
        Checks for Multicollinearity.
        Rule of Thumb: VIF > 5 or 10 indicates high correlation.
        """
        print("\n🔍 Checking VIF (Variance Inflation Factor)...")
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        
        # Calculate VIF for each feature
        # We perform this on the numeric matrix
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                           for i in range(len(X.columns))]
        
        print(vif_data.sort_values(by="VIF", ascending=False))
        return vif_data

    def run_production_pipeline(self):
        """
        The Full "Deep ML" Pipeline: 
        Log-Transform -> One-Hot -> VIF Check -> Scaling -> SGD -> Evaluation
        """
        print("\n🚀 Starting Production Pipeline...")
        
        # 1. Log Transformations (Economics)
        self.df['log_qty'] = np.log1p(self.df['quantity_sold'])
        self.df['log_price'] = np.log1p(self.df['price'])
        self.df['log_freight'] = np.log1p(self.df['freight_value'])
        
        # 2. Seasonality
        self.df['date_only'] = pd.to_datetime(self.df['date_only'])
        self.df['month'] = self.df['date_only'].dt.month
        self.df['is_weekend'] = (self.df['date_only'].dt.dayofweek >= 5).astype(int)
        
        # 3. One-Hot Encoding
        data = pd.get_dummies(self.df, columns=['product_category_name'], drop_first=True)
        
        # 4. Prepare X and y
        features_to_drop = ['date_only', 'quantity_sold', 'price', 'freight_value', 'log_qty']
        X = data.drop(columns=features_to_drop, errors='ignore').fillna(0)
        y = data['log_qty']
        
        # 5. VIF Check (Before Scaling)
        # We take a sample for VIF speed if data is huge
        self.check_vif(X.select_dtypes(include=[np.number]).sample(5000))
        
        # 6. Split & Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 7. Train SGD (Elastic Net for Feature Selection)
        print("\n🧠 Training SGD Regressor (Huber Loss for Robustness)...")
        model = SGDRegressor(
            loss='huber',          # Robust to outliers (better than squared_error)
            penalty='elasticnet',  # Mix of L1 (Lasso) and L2 (Ridge)
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=5000,
            learning_rate='adaptive',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # 8. Evaluation (Comprehensive Error Functions)
        preds = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   RMSE (Root Mean Sq Error): {rmse:.4f}")
        print(f"   MAE (Mean Abs Error):      {mae:.4f}")
        print(f"   R2 Score:                  {r2:.4f}")
        
        self.best_model = model
        self.best_scaler = self.scaler
        
        return model, self.scaler