import pandas as pd
from sklearn.utils.multiclass import type_of_target
import streamlit as st
import time

# Configuration
st.set_page_config(
    page_title="InsightFlow - Interactive ML Trainer",
    page_icon="🤖",
)

# Available models
CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Random Forest Classifier",
    "Multinomial Naive Bayes",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Random Forest Regressor",
    "Gaussian Naive Bayes"
]


if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "input_df" not in st.session_state:
    st.session_state.input_df = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None


def training_toast_handler(model_name: str = None):
    """
    Display a toast message during model training.

    Parameters:
    - model_name (str): Optional model name for context.
    """
    msg = st.toast('Train test split...')
    time.sleep(1)
    msg.toast('Training model...')
    time.sleep(2)
    msg.toast(
        f"Model `{model_name}` "
        "trained successfully!"
    )

def main():
    """Main application workflow"""
    st.title("🚀 InsightFlow: Smart Model Recommender")
    st.markdown("""
    ### Upload your dataset and get automatic ML suggestions!
    1. Upload a CSV file
    2. Select features & target
    3. Get model recommendations
    """)

    with st.form(key="data_upload_form"):
        # File upload section
        st.session_state.uploaded_file = st.file_uploader(
            "Choose a CSV dataset",
            type=["csv"],
            help="Maximum file size: 200MB"
        )

        # Form submission
        st.session_state.submitted = st.form_submit_button("Analyze Dataset")

    if st.session_state.submitted and st.session_state.uploaded_file:
        try:
            st.session_state.input_df = pd.read_csv(
                st.session_state.uploaded_file
            )
            st.success("✅ Dataset loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    if st.session_state.input_df is not None:
        with st.container(border=True, key="data_analysis_container"):
            st.subheader("📊 Dataset Analysis & model selection")
            # Data preview
            with st.expander("Preview first 10 rows"):
                st.dataframe(
                    st.session_state.input_df.head(10),
                    use_container_width=True
                )

            input_df_columns = st.session_state.input_df.columns.tolist()

            # Feature selection
            st.session_state.features = st.multiselect(
                "Select features for modeling",
                options=input_df_columns
            )

            if len(st.session_state.features) > 0:
                # Target selection
                target_options = [
                    col for col in input_df_columns
                    if col not in st.session_state.features
                ]
                st.session_state.target = st.selectbox(
                    "Select target variable",
                    options=target_options
                )

                if st.session_state.target:
                    # Task type detection
                    y = st.session_state.input_df[
                        st.session_state.target
                        ].dropna()
                    task_type = "classification" if "class" in type_of_target(y) \
                        else "regression"

                    # Model recommendations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"**Task Type:** `{task_type.capitalize()}`"
                        )

                    with col2:
                        if task_type == "classification":
                            models = CLASSIFICATION_MODELS
                        elif task_type == "regression":
                            models = REGRESSION_MODELS
                        else:
                            models = []

                        if models:
                            st.session_state.selected_model = st.selectbox(
                                "Available Models",
                                options=models
                            )
                        else:
                            st.warning("Unsupported task type")

                # Launch model training
                if st.session_state.selected_model:
                    if st.button("Train Model"):
                        training_toast_handler(
                            model_name=st.session_state.selected_model
                        )


if __name__ == "__main__":
    main()
