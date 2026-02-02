import pandas as pd
import os

class OlistDataEngine:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.datasets = {}

    def get_paths(self):
        """Verifies all 9 datasets exist."""
        required_files = [
            'olist_orders_dataset.csv',
            'olist_order_items_dataset.csv',
            'olist_products_dataset.csv',
            'olist_customers_dataset.csv',
            'olist_sellers_dataset.csv',
            'olist_order_reviews_dataset.csv',
            'olist_order_payments_dataset.csv',
            'olist_geolocation_dataset.csv',
            'product_category_name_translation.csv'
        ]
        
        missing = [f for f in required_files if not os.path.exists(os.path.join(self.data_path, f))]
        if missing:
            raise FileNotFoundError(f"❌ Missing Datasets: {missing}")
        print("✅ All 9 Datasets Found.")

    def load_data(self):
        """Loads all CSVs into a dictionary."""
        print("⏳ Loading all 9 datasets... (This may take a moment)")
        file_map = {
            'orders': 'olist_orders_dataset.csv',
            'items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'customers': 'olist_customers_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'reviews': 'olist_order_reviews_dataset.csv',
            'payments': 'olist_order_payments_dataset.csv',
            'geo': 'olist_geolocation_dataset.csv',
            'translate': 'product_category_name_translation.csv'
        }
        
        for key, filename in file_map.items():
            self.datasets[key] = pd.read_csv(os.path.join(self.data_path, filename))
        
        print("✅ Data Loaded Successfully.")
        return self.datasets

    def construct_master_table(self):
        """
        Joins all tables into one massive flat file (The Master Table).
        Logic: Items -> Orders -> Products -> Reviews -> Payments -> Customers -> Sellers
        """
        print("🏗️ Constructing Master Table (Star Schema Join)...")
        
        # 1. Start with ITEMS (The most granular level)
        df = self.datasets['items']
        
        # 2. Join ORDERS (Get timestamps and status)
        df = df.merge(self.datasets['orders'], on='order_id', how='left')
        
        # 3. Join PRODUCTS (Get category and specs)
        df = df.merge(self.datasets['products'], on='product_id', how='left')
        
        # 4. Join TRANSLATIONS (English Category Names)
        trans = self.datasets['translate']
        df = df.merge(trans, on='product_category_name', how='left')
        df['product_category_name'] = df['product_category_name_english'].fillna(df['product_category_name'])
        df.drop(columns=['product_category_name_english'], inplace=True)
        
        # 5. Join REVIEWS (Aggregate multiple reviews per order to one score)
        # An order might have 2 reviews. We take the average or first.
        reviews = self.datasets['reviews'].groupby('order_id').agg({
            'review_score': 'mean'
        }).reset_index()
        df = df.merge(reviews, on='order_id', how='left')
        
        # 6. Join PAYMENTS (Aggregate installments/value)
        # An order can have multiple payments (voucher + credit card).
        payments = self.datasets['payments'].groupby('order_id').agg({
            'payment_installments': 'max', # Max installments used
            'payment_value': 'sum',        # Total value
            'payment_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown' # Primary method
        }).reset_index()
        df = df.merge(payments, on='order_id', how='left')
        
        # 7. Join CUSTOMERS (Get Customer State)
        df = df.merge(self.datasets['customers'], on='customer_id', how='left')
        
        # 8. Join SELLERS (Get Seller State)
        df = df.merge(self.datasets['sellers'], on='seller_id', how='left')
        
        # 9. Geo-Location (Optional - usually too granular, we use State instead)
        # We skip merging raw lat/long per row to avoid exploding memory, 
        # relying on 'customer_state' and 'seller_state' instead.
        
        print(f"✅ Master Table Constructed. Shape: {df.shape}")
        return df