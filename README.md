# InsightFlow: Interactive ML Explorer & Trainer

**InsightFlow** is a user-friendly Streamlit app that allows you to:
- Upload your own CSV dataset
- Select features and target variables
- Automatically detect if your task is classification or regression
- Get tailored model recommendations
- (Coming soon) Train models and visualize results interactively

![InsightFlow Demo](demo/insight-flow-demo.gif)

---

## Features

- **CSV Uploader:** Easily upload and preview your data.
- **Feature Selection:** Choose which columns to use as features and target.
- **Task Detection:** The app analyzes your target and tells you whether your problem is classification or regression.
- **Model Suggestions:** Get a list of suitable machine learning models for your task.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/valeriova96/InsightFlow.git
cd insightflow
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

### 3. Activate the Environment

```bash
conda activate insight-flow
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).
