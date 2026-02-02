import pandas as pd
import numpy as np

class FeatureGenerator:
    def __init__(self, master_df):
        self.df = master_df.copy()

    def preprocess_and_engineer(self):
        """
        Cleaning -> Type Conversion -> Feature Extraction -> Aggregation
        """
        print("⚙️ Engineering Features (Production Level)...")
        
        # --- 1. DATETIME CONVERSIONS ---
        date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                     'order_delivered_carrier_date', 'order_delivered_customer_date', 
                     'order_estimated_delivery_date']
        
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
        # --- 2. LOGISTICS FEATURES (The "Hidden" Cost of E-commerce) ---
        # Actual Delivery Time (Days)
        self.df['delivery_days'] = (self.df['order_delivered_customer_date'] - self.df['order_purchase_timestamp']).dt.days
        
        # Estimated vs Actual (Is it late?)
        self.df['delay_days'] = (self.df['order_delivered_customer_date'] - self.df['order_estimated_delivery_date']).dt.days
        self.df['is_late'] = (self.df['delay_days'] > 0).astype(int)
        
        # --- 3. PRODUCT SPECS (Volume & Density) ---
        # Shipping cost is often driven by volume (LxWxH)
        self.df['product_volume_cm3'] = (self.df['product_length_cm'] * self.df['product_height_cm'] * self.df['product_width_cm'])
        
        # --- 4. GEOGRAPHY (Distance Proxy) ---
        # Are buyer and seller in the same state? (Cheaper shipping?)
        self.df['same_state'] = (self.df['customer_state'] == self.df['seller_state']).astype(int)
        
        # --- 5. AGGREGATION (Granular -> Daily Product Demand) ---
        # We must group by Product + Date to model Demand
        self.df['date_only'] = self.df['order_purchase_timestamp'].dt.date
        
        # What we want to predict: Quantity Sold per Day per Product
        # What explains it: Price, Avg Review, Avg Delivery Time, Seasonality
        
        daily_demand = self.df.groupby(['product_category_name', 'date_only']).agg({
            'order_item_id': 'count',       # Quantity Sold (Target)
            'price': 'mean',                # Avg Price that day
            'freight_value': 'mean',        # Avg Shipping Cost
            'review_score': 'mean',         # Reputation
            'delivery_days': 'mean',        # Service Speed
            'payment_installments': 'mean', # Affordability
            'same_state': 'mean',           # Local preference
            'is_late': 'mean'               # Quality Control
        }).reset_index()
        
        daily_demand.rename(columns={'order_item_id': 'quantity_sold'}, inplace=True)
        
        # --- 6. CLEANING ---
        daily_demand = daily_demand.dropna() # Drop rows where we lack critical data
        
        print(f"✅ Features Engineered. Final Dataset Shape: {daily_demand.shape}")
        return daily_demand