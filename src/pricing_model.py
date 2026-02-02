import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class PriceElasticityModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.coefficients = None
        
    def prepare_data(self):
        """
        Applies Log-Log transformation for Elasticity calculation.
        """
        print("Feature Engineering for Model...")
        
        # 1. Handle Zeros: Log(0) is -inf. We add 1 to avoid this (Log1p)
        # We model log(Quantity + 1)
        self.data['log_quantity'] = np.log1p(self.data['quantity_sold'])
        
        # 2. Log(Price)
        self.data['log_price'] = np.log1p(self.data['price'])
        
        # 3. One-Hot Encoding for Categories
        # This allows each category to have its own baseline demand
        self.data = pd.get_dummies(self.data, columns=['product_category_name'], drop_first=True)
        
        # Drop original columns we don't need for regression
        # We keep 'log_price', 'log_quantity', 'month', 'is_weekend'
        drop_cols = ['product_id', 'date_only', 'quantity_sold', 'price', 'order_purchase_timestamp']
        self.data = self.data.drop(columns=drop_cols, errors='ignore')
        
        # Handle any NaNs created
        self.data = self.data.fillna(0)
        
        print(f"Model Data Shape: {self.data.shape}")

    def train(self):
        """
        Trains the Linear Regression Model on Log-Transformed Data
        """
        print("🚀 Training Model...")
        
        # X = All columns except log_quantity
        X = self.data.drop(columns=['log_quantity'])
        # y = log_quantity
        y = self.data['log_quantity']
        
        # Split (Standard 80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit Linear Regression
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        r2 = r2_score(y_test, preds)
        print(f"✅ Model Trained. R2 Score: {r2:.4f}")
        
        # Save coefficients for analysis
        self.coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': self.model.coef_
        })
        
        return self.model

    def get_elasticity(self):
        """
        Returns the specific coefficient for Price.
        Since it's a log-log model, this IS the elasticity.
        """
        price_coef = self.coefficients[self.coefficients['Feature'] == 'log_price']['Coefficient'].values[0]
        return price_coef