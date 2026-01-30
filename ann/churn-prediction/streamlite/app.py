import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("🔮 Customer Churn Prediction")
st.markdown("Predict customer churn probability using Artificial Neural Network")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model = st.sidebar.selectbox("Model Version", options=["v1"], index=0)

# Main form
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Customer's age in years")
    gender = st.selectbox("Gender", options=["Male", "Female"], help="Customer's gender")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, help="How long the customer has been with the company")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=5.0, help="Monthly bill amount")

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0, help="Total amount billed to customer")
    contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], help="Type of customer contract")
    payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"], help="How customer pays their bill")

# Validation
if total_charges < monthly_charges:
    st.warning("⚠️ Total charges should typically be greater than or equal to monthly charges")

# Prediction button
if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        try:
            # Create input DataFrame with proper format (single row)
            input_data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "Contract": [contract],
                "PaymentMethod": [payment_method],
                "TotalCharges": [total_charges]
            })
            
            # Build absolute model path
            model_dir = PROJECT_ROOT / "models" / model
            
            if not model_dir.exists():
                st.error(f"❌ Model directory not found: {model_dir}")
                st.info("Please ensure the model is trained first by running: `python run.py train`")
            else:
                # Make prediction
                probs, preds = predict(input_data, str(model_dir))
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                probability = probs[0]
                prediction = preds[0]
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Prediction", "Yes" if prediction == 1 else "No")
                
                with col2:
                    st.metric("Churn Probability", f"{probability:.1%}")
                
                with col3:
                    st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                
                # Visual indicator
                if probability >= 0.5:
                    st.error(f"⚠️ **HIGH RISK**: This customer is likely to churn (Probability: {probability:.1%})")
                    st.markdown("**Recommended Actions:**")
                    st.markdown("- Reach out with retention offers")
                    st.markdown("- Review customer satisfaction")
                    st.markdown("- Consider contract upgrade incentives")
                else:
                    st.success(f"✅ **LOW RISK**: This customer is likely to stay (Probability: {probability:.1%})")
                    st.markdown("**Recommended Actions:**")
                    st.markdown("- Maintain current service level")
                    st.markdown("- Consider upselling opportunities")
                
                # Progress bar for visualization
                st.markdown("### Churn Risk Level")
                st.progress(float(probability))
                
        except FileNotFoundError as e:
            st.error(f"❌ Error: Model files not found - {e}")
            st.info("Please train the model first using: `python run.py train`")
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("**Model Info:** ANN-based churn prediction with dropout regularization")
st.markdown("*Developed using TensorFlow/Keras and deployed with Streamlit*")
