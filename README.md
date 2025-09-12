# AI-Powered Crop Yield Prediction System

This project is a smart crop yield prediction system designed for Indian farmers, built for a hackathon. It uses machine learning to predict crop yields based on historical data and environmental factors.

## Features
- **Yield Prediction:** Predicts yield for Rice, Wheat, and Maize using a Random Forest model.
- **Data-Driven Recommendations:** Provides insights into factors affecting yield.
- **Risk Alerts:** (Conceptual) Can be extended to alert for drought, pests, etc.
- **Interactive Dashboard:** An easy-to-use Streamlit dashboard for farmers to get predictions.
- **AI Explainability:** Uses SHAP to explain why a prediction was made.

## Tech Stack
- **Python:** Pandas, NumPy, Scikit-learn
- **Dashboard:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Explainability:** SHAP
- **Deployment (Optional):** Docker, Streamlit Community Cloud

## Folder Structure
crop-yield-prediction/
│
├── 📂 data/
├── 📂 dashboard/
├── 📂 models/
├── 📂 notebooks/
├── 📂 src/
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 setup.py


## Setup & Usage

1.  **Clone the repository:**
    git clone <your-repo-url>
    cd crop-yield-prediction

2.  **Create a virtual environment and activate it:**
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  **Install the dependencies:**
    pip install -r requirements.txt

4.  **Run the training pipeline:**
    python src/pipeline/train_pipeline.py

5.  **Run the Streamlit dashboard:**
    streamlit run dashboard/app.py