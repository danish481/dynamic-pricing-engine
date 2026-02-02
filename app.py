import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# --- 1. LOAD THE TRAINED ARTIFACTS ---
# We use caching so it doesn't reload on every slider move
@st.cache_resource
def load_artifacts():
    model = joblib.load('src/best_model.pkl')
    scaler = joblib.load('src/best_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run 'main.py' first to generate best_model.pkl")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="TiVTO Pricing Engine (Deep ML)", page_icon="🧠", layout="wide")

st.title("🧠 TiVTO: AI-Powered Pricing Engine")
st.markdown("**Powered by SGDRegressor (ElasticNet) • Trained on 110k Orders**")

# --- SIDEBAR: SCENARIO CONTROLS ---
st.sidebar.header("🎛️ Simulation Parameters")

# We need "Default" values for features that the user ISN'T changing
# These are the "Average" values from your dataset
defaults = {
    'price': 120.00,
    'freight_value': 20.00,
    'review_score': 4.5,
    'payment_installments': 4,
    'delivery_days': 12,
    'is_late': 0,
    'same_state': 0,
    'is_weekend': 0,
    'month': 8 # August (Average month)
}

# 1. Price Slider (The Main Lever)
base_price = st.sidebar.number_input("Base Product Price ($)", value=defaults['price'])
price_change_pct = st.sidebar.slider("Adjust Price (%)", -50, 50, 0, 1) / 100.0

# 2. Advanced Controls (Context)
with st.sidebar.expander("Show Advanced Levers"):
    st.write("Does improving service justify a higher price?")
    sim_freight = st.slider("Shipping Cost ($)", 0.0, 100.0, defaults['freight_value'])
    sim_score = st.slider("Review Score (1-5)", 1.0, 5.0, defaults['review_score'])
    sim_delivery = st.slider("Delivery Days", 1, 60, defaults['delivery_days'])

# --- PREDICTION ENGINE ---
def predict_demand(price_input, freight_input, score_input, delivery_input):
    """
    Constructs a single feature row and scales it for prediction.
    Must match the EXACT column order used in training!
    """
    # 1. Feature Engineering (Same logic as training)
    log_price = np.log1p(price_input)
    log_freight = np.log1p(freight_input)
    
    # 2. Create DataFrame with correct columns
    # Note: We need dummy variables for categories. For simplicity in simulation,
    # we assume "Other" category (all zeros for category columns)
    # If you want specific categories, we'd need a dropdown.
    
    # Based on your VIF output, the scaler expects specific numeric columns.
    # We construct the input vector.
    
    # CRITICAL: This list must match X.columns from your experiments.py
    # Since we used get_dummies, there are many columns.
    # We will initialize a Zero vector and fill what we know.
    
    input_data = pd.DataFrame([{
        'review_score': score_input,
        'delivery_days': delivery_input,
        'payment_installments': defaults['payment_installments'],
        'same_state': defaults['same_state'],
        'is_late': 0, # Assume on time
        'log_price': log_price,
        'log_freight': log_freight,
        'month': defaults['month'],
        'is_weekend': defaults['is_weekend']
        # Categories are missing here, essentially treating as "Baseline Category"
    }])
    
    # Align with Scaler's expected features
    # (The scaler was fitted on X_train, so it knows the feature names)
    try:
        # We need to ensure input_data has ALL columns scaler saw.
        # This is a common ML Ops challenge.
        # Quick Fix: Get feature names from the scaler if possible, or pad with 0s.
        required_features = scaler.feature_names_in_
        
        # Add missing columns (categories) as 0
        for col in required_features:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Reorder columns to match training
        input_data = input_data[required_features]
        
    except AttributeError:
        st.error("Scaler version mismatch. Please re-run main.py")
        return 0

    # 3. Scale & Predict
    scaled_input = scaler.transform(input_data)
    log_pred = model.predict(scaled_input)[0]
    
    # 4. Inverse Log (Get actual units)
    return np.expm1(log_pred)

# --- CALCULATE SCENARIOS ---
# 1. Current State
current_price = base_price * (1 + price_change_pct)
pred_demand = predict_demand(current_price, sim_freight, sim_score, sim_delivery)
pred_revenue = current_price * pred_demand

# 2. Baseline (0% change)
base_demand = predict_demand(base_price, sim_freight, sim_score, sim_delivery)
base_revenue = base_price * base_demand

rev_lift = pred_revenue - base_revenue

# --- DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Simulated Price", f"${current_price:,.2f}", f"{price_change_pct:.0%}")
col2.metric("Predicted Daily Sales", f"{pred_demand:.1f} Units", f"{pred_demand - base_demand:.1f}")
col3.metric("Proj. Daily Revenue", f"${pred_revenue:,.2f}", f"${rev_lift:,.2f}", delta_color="normal")

# --- REVENUE CURVE ---
st.divider()
st.subheader("📉 Revenue Optimization Curve")

x_range = np.linspace(-0.5, 0.5, 50)
y_revs = []

for x in x_range:
    p = base_price * (1 + x)
    d = predict_demand(p, sim_freight, sim_score, sim_delivery)
    y_revs.append(p * d)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_range, y=y_revs, mode='lines', name='Revenue', line=dict(color='#00CC96', width=4)))
fig.add_trace(go.Scatter(x=[price_change_pct], y=[pred_revenue], mode='markers', name='You are Here', marker=dict(color='red', size=15)))

fig.update_layout(xaxis_title="Price Change (%)", yaxis_title="Revenue ($)", xaxis_tickformat='.0%')
st.plotly_chart(fig, use_container_width=True)

# --- MODEL INTERPRETATION ---
st.info(f"""
**Model Insight:**
This prediction is running live on `SGDRegressor`.
Notice how changing **Shipping Cost** or **Reviews** in the advanced panel shifts the entire revenue curve?
That is the power of Multivariate Regression.
""")