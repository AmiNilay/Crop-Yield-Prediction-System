import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import os

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

# --- Caching Functions ---
@st.cache_resource
def load_prediction_pipeline():
    """Loads the prediction pipeline and checks for model files."""
    model_path = 'models/model.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error("Model files not found. Please run the training pipeline first by executing 'python src/pipeline/train_pipeline.py' in your terminal.")
        st.stop()
    return PredictPipeline()

@st.cache_data
def load_unique_values():
    """Loads unique states and crops from the data file safely."""
    filepath = "data/raw/crop_production_india.csv"
    if not os.path.exists(filepath):
        return [], []
    df = pd.read_csv(filepath)
    states = sorted(df.dropna(subset=['State'])['State'].astype(str).unique().tolist())
    crops = sorted(df.dropna(subset=['Crop'])['Crop'].astype(str).unique().tolist())
    return states, crops

# --- Initialization ---
try:
    predict_pipeline = load_prediction_pipeline()
    model = predict_pipeline.model
    preprocessor = predict_pipeline.preprocessor
    states, crops = load_unique_values()
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()

# --- Main Application ---
st.title("🌾 AI-Powered Crop Yield Prediction System")
st.write("Predicts yield based on crop, state, and cost of cultivation data.")
st.markdown("---")

# --- Sidebar for Input ---
st.sidebar.header("Enter Cultivation Details")

if not states or not crops:
    st.sidebar.error("Could not load crop or state data. Please check 'data/raw/crop_production_india.csv'.")
else:
    crop = st.sidebar.selectbox("Select Crop", crops)
    state = st.sidebar.selectbox("Select State", states)
    cost_a2_fl = st.sidebar.number_input("Cost of Cultivation (A2+FL, ₹/Hectare)", min_value=5000.0, value=17000.0, step=100.0)
    cost_c2 = st.sidebar.number_input("Cost of Cultivation (C2, ₹/Hectare)", min_value=8000.0, value=25000.0, step=100.0)
    cost_prod_c2 = st.sidebar.number_input("Cost of Production (C2, ₹/Quintal)", min_value=90.0, value=2000.0, step=50.0)

    if st.sidebar.button("Predict Yield"):
        assert crop is not None, "Crop selection cannot be None"
        assert state is not None, "State selection cannot be None"

        with st.spinner("Analyzing your data..."):
            data = CustomData(
                crop=crop,
                state=state,
                cost_of_cultivation_a2_fl=cost_a2_fl,
                cost_of_cultivation_c2=cost_c2,
                cost_of_production_c2=cost_prod_c2
            )
            pred_df = data.get_data_as_dataframe()
            
            prediction = predict_pipeline.predict(pred_df)
            
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.subheader("Prediction Result")
                st.metric(label="Predicted Yield (Quintal/Hectare)", value=f"{prediction[0]:.2f}")

            with res_col2:
                st.subheader("Prediction Explanation")
                st.write("This chart shows how each factor contributed to the final prediction.")
                
                transformed_data = preprocessor.transform(pred_df)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(transformed_data)
                
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    cat_features = preprocessor.named_transformers_['cat_pipeline']['onehot'].get_feature_names_out(['crop', 'state'])
                    num_features = ['cost_of_cultivation_hectare_a2_fl', 'cost_of_cultivation_hectare_c2', 'cost_of_production_quintal_c2']
                    feature_names = list(num_features) + list(cat_features)

                # --- THIS IS THE FIX ---
                # 1. Create the SHAP plot, which draws on the current Matplotlib figure
                shap.force_plot(explainer.expected_value, shap_values[0], feature_names, matplotlib=True, show=False)
                
                # 2. Get the current figure using plt.gcf()
                fig = plt.gcf()
                fig.tight_layout() # Adjust layout to prevent labels overlapping

                # 3. Pass the figure object explicitly to st.pyplot()
                st.pyplot(fig)

                # 4. Clear the current figure to ensure the next plot starts fresh
                plt.clf()
                # --- END OF FIX ---