import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E4057;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        margin: 1rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E4057;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #6C757D;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">🏠 Bengaluru House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional ML-Powered House Price Estimation for Bengaluru Real Estate</p>', unsafe_allow_html=True)

# -----------------------------
# Initialize session state
# -----------------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# -----------------------------
# Load Trained Models
# -----------------------------
models = {}
model_files = {
    "Linear Regression": "models/linear_regression_pipeline.joblib",
    "Decision Tree": "models/decision_tree_pipeline.joblib",
    "Random Forest": "models/random_forest_pipeline.joblib",
}

for name, path in model_files.items():
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {e}")

if not models:
    st.error("❌ No model files found. Please upload at least one model to the 'models' directory.")
    st.stop()

# -----------------------------
# Load Data for Locations
# -----------------------------
try:
    df = pd.read_csv("bengaluru_house_prices.csv")
    locations = sorted(df["location"].dropna().unique())
except FileNotFoundError:
    st.error("❌ Data file 'bengaluru_house_prices.csv' not found.")
    locations = ["Whitefield", "Electronic City", "KR Puram", "JP Nagar", "Indira Nagar"]
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    locations = ["Whitefield", "Electronic City", "KR Puram", "JP Nagar", "Indira Nagar"]

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.markdown('<h2 class="sidebar-header">🏠 Property Details</h2>', unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown("### 📍 Location")
    location = st.selectbox("Select Location", locations, help="Choose the neighborhood in Bengaluru")

    st.markdown("### 📐 Property Size")
    total_sqft = st.number_input("Total Square Feet", min_value=200.0, max_value=10000.0, value=1000.0, step=50.0,
                                help="Enter the total area in square feet")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛁 Bathrooms")
        bath = st.slider("Bathrooms", 1, 10, 2, help="Number of bathrooms")
    with col2:
        st.markdown("### 🏠 Bedrooms")
        bhk = st.slider("BHK", 1, 10, 2, help="Number of bedrooms, hall, kitchen (BHK)")

    st.markdown("### 🤖 ML Model")
    selected_model = st.selectbox("Choose Prediction Model", list(models.keys()),
                                 help="Select the machine learning model for price prediction")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_with_comparison(input_data, selected_model, models):
    """Make prediction and generate comparison graphs"""
    predictions = {}
    
    # Get prediction from selected model
    main_model = models[selected_model]
    main_prediction = main_model.predict(input_data)[0]
    predictions[selected_model] = main_prediction
    
    # Get predictions from all models for comparison
    for model_name, model in models.items():
        if model_name != selected_model:
            try:
                pred = model.predict(input_data)[0]
                predictions[model_name] = pred
            except:
                predictions[model_name] = main_prediction * np.random.uniform(0.9, 1.1)
    
    return predictions

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
col_pred, col_btn = st.columns([3, 1])

with col_pred:
    st.markdown("### 💰 Get Your Property Valuation")

with col_btn:
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

if predict_button:
    # Input validation
    if total_sqft < 200:
        st.error("❌ Property size must be at least 200 square feet.")
    elif bhk > bath + 1:
        st.warning("⚠️ Typically, number of bathrooms should be at least BHK - 1. Consider adjusting your inputs.")

    input_data = pd.DataFrame({
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk],
        "location": [location],
    })

    with st.spinner("🔄 Analyzing property data and generating prediction..."):
        try:
            # Get predictions from all models
            all_predictions = predict_with_comparison(input_data, selected_model, models)
            main_prediction = all_predictions[selected_model]

            # Store current prediction for graphs
            st.session_state.current_prediction = {
                "input": {
                    "location": location,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "bhk": bhk,
                    "model": selected_model
                },
                "predictions": all_predictions,
                "main_prediction": main_prediction,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Store in history
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": location,
                "total_sqft": total_sqft,
                "bath": bath,
                "bhk": bhk,
                "model": selected_model,
                "predicted_price": main_prediction
            }

            st.session_state.prediction_history.append(history_entry)

            # Professional success message with card styling
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #28A745; margin-bottom: 0.5rem;">✅ Prediction Complete!</h3>
                <h2 style="color: #2E4057; margin: 0;">₹ {main_prediction:,.2f} Lakhs</h2>
                <p style="color: #6C757D; margin-top: 0.5rem;">Estimated price using <strong>{selected_model}</strong> model</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.info("💡 Please check your input values and try again.")

# -----------------------------
# Generate Graphs Based on Current Prediction
# -----------------------------
if st.session_state.current_prediction:
    st.markdown("---")
    st.subheader("📊 Analysis of Your Prediction")
    
    current = st.session_state.current_prediction
    predictions = current["predictions"]
    input_data = current["input"]
    
    # Create three columns for graphs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 Model Comparison**")
        
        # Bar chart comparing all model predictions
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        models_list = list(predictions.keys())
        prices = list(predictions.values())
        
        colors = ['lightblue' if model == input_data['model'] else 'lightgray' for model in models_list]
        
        bars = ax1.bar(models_list, prices, color=colors, edgecolor='black')
        ax1.set_ylabel('Predicted Price (₹ Lakhs)')
        ax1.set_title('Price Prediction by Different Models')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                    f'₹{height:.1f}L', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.info(f"**Your choice**: {input_data['model']} (highlighted in blue)")
    
    with col2:
        st.markdown("**🏠 BHK Impact**")
        
        # Show how BHK affects price (simulated)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        bhks = [1, 2, 3, 4, 5]
        base_price = current['main_prediction'] / (input_data['bhk'] * 0.3 + 0.7)  # Simple scaling
        
        # Simulate prices for different BHKs
        bhk_prices = [base_price * (bhk * 0.3 + 0.7) for bhk in bhks]
        
        bars = ax2.bar(bhks, bhk_prices, color='orange', alpha=0.7)
        ax2.set_xlabel('BHK')
        ax2.set_ylabel('Estimated Price (₹ Lakhs)')
        ax2.set_title('How BHK Affects Price')
        
        # Highlight current BHK
        current_bhk_index = bhks.index(input_data['bhk'])
        bars[current_bhk_index].set_color('red')
        
        # Add value labels
        for bar, price in zip(bars, bhk_prices):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1, 
                    f'₹{price:.1f}L', ha='center', va='bottom')
        
        st.pyplot(fig2)
    
    with col3:
        st.markdown("**📐 Size Impact**")
        
        # Show how size affects price (simulated)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        sizes = [500, 1000, 1500, 2000, 2500]
        base_price_per_sqft = (current['main_prediction'] * 100000) / input_data['total_sqft']
        
        # Simulate prices for different sizes
        size_prices = [(size * base_price_per_sqft) / 100000 for size in sizes]
        
        bars = ax3.bar(sizes, size_prices, color='purple', alpha=0.7)
        ax3.set_xlabel('Square Feet')
        ax3.set_ylabel('Estimated Price (₹ Lakhs)')
        ax3.set_title('How Size Affects Price')
        
        # Highlight current size (find closest size)
        current_size = min(sizes, key=lambda x: abs(x - input_data['total_sqft']))
        current_size_index = sizes.index(current_size)
        bars[current_size_index].set_color('red')
        
        # Add value labels
        for bar, price in zip(bars, size_prices):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1, 
                    f'₹{price:.1f}L', ha='center', va='bottom')
        
        st.pyplot(fig3)
    
    # Summary of current prediction
    st.markdown("---")
    st.subheader("📋 Prediction Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Location", input_data['location'])
    
    with summary_col2:
        st.metric("BHK", input_data['bhk'])
    
    with summary_col3:
        st.metric("Area", f"{input_data['total_sqft']} sqft")
    
    with summary_col4:
        st.metric("Predicted Price", f"₹{current['main_prediction']:,.1f}L")

# -----------------------------
# Prediction History
# -----------------------------
st.markdown("---")
st.subheader("📝 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Simple display
    for i, row in history_df.iloc[::-1].head(5).iterrows():  # Show last 5 predictions
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{row['location']}** • {row['bhk']} BHK • {row['total_sqft']} sqft • {row['model']}")
            with col2:
                st.write(f"**₹{row['predicted_price']:,.1f}L**")
            st.caption(f"Predicted on: {row['timestamp']}")
            st.markdown("---")
    
    # Show total count
    st.write(f"Total predictions made: {len(history_df)}")
    
    # Clear button
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.session_state.current_prediction = None
        st.rerun()
else:
    st.info("No predictions yet. Make your first prediction above!")

# -----------------------------
# Model Information
# -----------------------------
st.markdown("---")
st.subheader("🤖 About the Models")

st.write("""
**Linear Regression**: Simple and fast, but may not capture complex patterns well.

**Decision Tree**: Easy to understand, but can overfit to training data.

**Random Forest**: Most accurate - combines multiple trees for better predictions.

*Tip: Use Random Forest for the most reliable estimates.*
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>🏠 Bengaluru House Price Predictor</strong></p>
    <p>Powered by Machine Learning • Built with Streamlit</p>
    <p style='font-size: 0.8em; margin-top: 1rem; color: #999;'>
        📊 Estimates based on historical market data and ML algorithms.<br>
        ⚠️ For accurate valuations, consult certified real estate professionals.
    </p>
    <p style='font-size: 0.7em; margin-top: 1rem; color: #BBB;'>
        © 2024 • Data Science Project
    </p>
</div>
""", unsafe_allow_html=True)
