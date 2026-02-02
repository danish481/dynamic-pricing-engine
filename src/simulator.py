import pandas as pd
import numpy as np

class PricingSimulator:
    def __init__(self, base_df, elasticity):
        """
        base_df: The data with 'quantity_sold' and 'price' (Average daily values)
        elasticity: The coefficient from our model (-0.0161)
        """
        self.df = base_df
        self.elasticity = elasticity
        
    def simulate_scenario(self, price_change_percent):
        """
        Simulates: If we change price by X%, what happens to Revenue?
        price_change_percent: e.g., 0.10 for +10%, -0.20 for -20%
        """
        print(f"\n🔮 Simulation: Changing Price by {price_change_percent*100}%...")
        
        # 1. Current State
        current_price = self.df['price'].mean()
        current_quantity = self.df['quantity_sold'].sum()
        current_revenue = (self.df['price'] * self.df['quantity_sold']).sum()
        
        # 2. Future State (The Economic Formula)
        # New Price = Old Price * (1 + change)
        new_price = current_price * (1 + price_change_percent)
        
        # New Demand = Old Demand * (1 + (Elasticity * %Change in Price))
        demand_change_percent = self.elasticity * price_change_percent
        new_quantity = current_quantity * (1 + demand_change_percent)
        
        # New Revenue (Approximation)
        new_revenue = new_price * new_quantity
        
        # 3. The Delta
        revenue_diff = new_revenue - current_revenue
        
        print(f"   📉 Demand Change: {demand_change_percent*100:.2f}%")
        print(f"   💰 Revenue Impact: ${revenue_diff:,.2f}")
        
        if revenue_diff > 0:
            print("   ✅ RECOMMENDATION: Green Light. Revenue increases.")
        else:
            print("   ❌ RECOMMENDATION: Stop. Revenue decreases.")