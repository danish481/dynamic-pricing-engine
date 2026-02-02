import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TiVTO Pricing Engine",
    page_icon="💸",
    layout="wide"
)

# --- HEADER ---
st.title("💸 Dynamic Pricing & Revenue Optimization Engine")
st.markdown("""
**Project:** Olist E-Commerce Revenue Simulator  
**Goal:** Determine the optimal price point to maximize total revenue based on ML-derived elasticity.
""")

st.divider()

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Simulation Parameters")

# 1. Base Stats (derived from your analysis)
base_price = st.sidebar.number_input("Average Product Price ($)", value=120.0, step=1.0)
base_volume = st.sidebar.number_input("Average Daily Sales Volume (Units)", value=500, step=10)
elasticity = st.sidebar.number_input("Price Elasticity (β)", value=-0.0161, step=0.001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Model Insight:** An elasticity of **{elasticity}** implies demand is **Inelastic**.  
    Customers are not sensitive to price changes.
    """
)

# --- MAIN INTERFACE: THE SLIDER ---
st.subheader("🔮 Interactive Scenario Simulator")

# FIX: Use Integers (-50 to +50) for the slider so it displays "10%" correctly
price_change_int = st.slider(
    "Adjust Price Change (%)", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=1,
    format="%d%%"
)

# Convert back to decimal for calculations (e.g., 10 becomes 0.10)
price_change_pct = price_change_int / 100.0

# --- CALCULATIONS ---
# 1. New Price
new_price = base_price * (1 + price_change_pct)

# 2. New Demand (Economic Formula)
demand_change_pct = elasticity * price_change_pct
new_volume = base_volume * (1 + demand_change_pct)

# 3. Revenue Comparison
current_revenue = base_price * base_volume
new_revenue = new_price * new_volume
revenue_lift = new_revenue - current_revenue

# --- METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("New Price", f"${new_price:,.2f}", f"{price_change_pct:.0%}")

with col2:
    st.metric("Predicted Demand", f"{new_volume:,.0f} units", f"{demand_change_pct:.2%}", delta_color="inverse")

with col3:
    st.metric("Projected Revenue", f"${new_revenue:,.2f}", f"${revenue_lift:,.2f}")

with col4:
    # Custom status logic
    if revenue_lift > 0:
        st.success(f"✅ Revenue Gain: +${revenue_lift:,.2f}")
    elif revenue_lift < 0:
        st.error(f"❌ Revenue Loss: -${abs(revenue_lift):,.2f}")
    else:
        st.warning("⚠️ No Change")

# --- CHARTING THE CURVE ---
st.divider()
st.subheader("📊 Revenue Optimization Curve")

# Create a range of scenarios to plot the curve
x_values = np.linspace(-0.5, 0.5, 100) # -50% to +50%
y_revenues = []

for x in x_values:
    # Calculate revenue for every possible price point
    sim_price = base_price * (1 + x)
    sim_vol = base_volume * (1 + (elasticity * x))
    y_revenues.append(sim_price * sim_vol)

# Find the Peak
max_rev_index = np.argmax(y_revenues)
optimal_pct = x_values[max_rev_index]

# Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_revenues, mode='lines', name='Projected Revenue', line=dict(color='#00CC96', width=4)))

# Add marker for CURRENT selection
fig.add_trace(go.Scatter(x=[price_change_pct], y=[new_revenue], mode='markers', name='Selected Scenario', marker=dict(color='red', size=15)))

fig.update_layout(
    xaxis_title="Price Change (%)",
    yaxis_title="Total Revenue ($)",
    xaxis_tickformat='.0%',
    yaxis_tickprefix='$',
    hovermode="x"
)

st.plotly_chart(fig, use_container_width=True)

# --- INSIGHTS ---
st.info(f"💡 **Strategy:** Based on the curve, the theoretical revenue maximum occurs at a price increase of **{optimal_pct:.0%}**.")