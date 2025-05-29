
# **InsightFlow: The Perfect Portfolio Project for Aspiring Data Scientists**

_Showcase your skills with pandas, scikit-learn, and streamlit in one interactive, reproducible app._

----------
![InsightFlow Demo](https://raw.githubusercontent.com/valeriova96/InsightFlow/main/demo/insight-flow-demo.gif)
----------

## **Introduction**

Breaking into data science isn’t easy. Recruiters want to see more than just a certificate—they want proof you can wrangle data, build models, and communicate results. But how do you build a project that’s impressive, practical, and reproducible, without getting lost in complexity?

InsightFlow is an interactive machine learning explorer and trainer built with the most popular data science libraries—**pandas**, **scikit-learn**, and **streamlit**. Whether you’re looking to boost your portfolio or practice end-to-end ML workflows, InsightFlow is the perfect launchpad.

----------

## **Why InsightFlow?**

Many beginner projects focus on a single dataset or algorithm. InsightFlow is different: it’s a general-purpose, user-friendly app that lets you upload any tabular dataset, select features and targets, and instantly get model recommendations and results. It’s designed to:

-   **Demonstrate real-world data science skills**  (data cleaning, EDA, model selection)
    
-   **Showcase best practices**  (reproducibility, documentation, modular code)
    
-   **Impress recruiters**  with a live, interactive demo
    

----------

## **Key Features**

-   **CSV Uploader:**  Bring your own data, or use classic datasets like Breast Cancer or Fish Market.
    
-   **Feature & Target Selection:**  Choose which columns to use for prediction.
    
-   **Automatic Task Detection:**  InsightFlow figures out if your problem is classification or regression.
    
-   **Model Recommendations:**  Get a curated list of models for your task.
    
-   **(Coming Soon) Model Training & Visualization:**  Train and evaluate models with one click.
    

----------

## **Tech Stack: The Data Science Trinity**

InsightFlow is built with three of the most essential Python libraries for data science:

## **1. Pandas**

Used for data loading, cleaning, and manipulation. If you want to be a data scientist, you need to master [pandas](https://pandas.pydata.org/).

## **2. Scikit-learn**

The go-to library for machine learning in Python. InsightFlow uses [scikit-learn](https://scikit-learn.org/stable/) for model selection, training, and evaluation.

## **3. Streamlit**

A modern, open-source framework for building beautiful data apps with minimal code. [Streamlit](https://streamlit.io/) powers InsightFlow’s interactive UI.

----------

## **How It Works**

1.  **Upload your CSV dataset**
    
2.  **Select features and target variable**
    
3.  **InsightFlow detects the task type**  (classification or regression)
    
4.  **Get model recommendations**
    
    -   _Classification:_  Logistic Regression, Random Forest, Support Vector Machine
        
    -   _Regression:_  Linear Regression, Random Forest Regressor, Support Vector Regression
        
5. **Train models and visualize results**  (metrics)
    

----------

## **Getting Started in Minutes**

**1. 🗂️ Clone the repo:**
```bash
git clone https://github.com/valeriova96/InsightFlow.git 
cd insightflow
``` 

**2. 🌏 Create the environment:**
```bash
conda env create -f environment.yml
conda activate insight-flow
```

**3. 🏃🏻 Run the app:**

`streamlit run app.py` 

**4. 🎮 Try it out!**  
Upload your own dataset, select your features, and see the magic happen.

----------

## **Why This Project Stands Out for Your Portfolio**
- **Proficiency**: Uses the most popular data science libraries—making it easy for recruiters to recognize and appreciate your skills.   
-   **Interactivity:**  Recruiters can see your work in action, not just in screenshots.
    
-   **Extensibility:**  The code is easy to expand—add new models, visualizations, or data sources as you grow.
    

----------
## **Features**
The main features of this project include automatic task recognition and model recommendation. Essentially, based on the chosen target column, we leverage the _scikit-learn_ function `type_of_target`. This function infers whether the problem is _binary classification_, _multiclass_, _multilabel_, or _regression_ based on the target column's values.
```python
CLASSIFICATION_TARGETS = [
	"binary",
	"multiclass",
	"multiclass-multioutput",
	"multilabel-indicator"
]

y = input_df[target]
task_type = "classification" if type_of_target(y) in CLASSIFICATION_TARGETS else "regression"
``` 

Once the task is identified, we can recommend a suitable ML model to use, but before digging into the training, it is important to preprocess the data a little bit.
The following function encapsulates this process and more.
```python
def train_and_evaluate_model(
        task_type: Literal["classification", "regression"],
        model_name: str,
        dataset: pd.DataFrame,
        feature_cols: list,
        target_col: str
) -> pd.DataFrame:
    # Clean data
    cleaned_data = clean_data(dataset, target_col)

    # Convert categorical columns to numerical codes
    if task_type == "regression":
        dataset = find_and_convert_cat_cols(cleaned_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        cleaned_data, feature_cols, target_col
    )

    metrics_df = pd.DataFrame()

    match task_type:
        case "classification":
            match model_name:
                case "Logistic Regression":
                    from models.classification.models import (
                        train_logistic_regression_model
                    )
                    model = train_logistic_regression_model(X_train, y_train)
                case "Random Forest Classifier":
                    from models.classification.models import (
                        train_random_forest_model
                    )
                    model = train_random_forest_model(X_train, y_train)
                case "Support Vector Machine":
                    from models.classification.models import (
                        train_support_vector_machine
                    )
                    model = train_support_vector_machine(X_train, y_train)
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            # Evaluate the model
            metrics_df = evaluate_class_model(model, X_test, y_test)

        case "regression":
            match model_name:
                case "Linear Regression":
                    from models.regression.models import (
                        train_linear_regression_model
                    )
                    model = train_linear_regression_model(X_train, y_train)
                case "Random Forest Regressor":
                    from models.regression.models import (
                        train_random_forest_model
                    )
                    model = train_random_forest_model(X_train, y_train)
                case "Support Vector Regression":
                    from models.regression.models import (
                        train_support_vector_regression
                    )
                    model = train_support_vector_regression(X_train, y_train)
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            metrics_df = evaluate_regr_model(model, X_test, y_test)

    return metrics_df
```
**NOTE**: These `case _:` blocks ensure that unpredicted elements raise a clear error rather than causing silent failures—making the system more robust.

**1. 🧹 Clean data**
First, we clean the data by calling the function `clean_data` that I developed. It is a very simple function that remove rows corresponding to _NaN_ or empty values. This helps prevent unexpected behavior or errors during model training.

**2. 🔄 Convert categorical data**
Since many regression algorithms cannot handle categorical data, we need to map those data into numbers that those algorithm can easily digest.

**3. 🖖🏻 Split data**
We simply split the dataset with the standard **scikit-learn** function.

**4. 🦾 Train and evaluate**
Now, we can finally train our model with the _train set_ and obtain some metrics for the _test set_. Thus, we create a dataframe where we plug into the metrics for the current ML task:
* precision
* recall
* F1-score

for `classification`,

* R2
* MSE

for `regression`.

Here’s an example of what the metrics dataframe might look like for a regression task:
| MSE      | R-squared |
|----------|-----------|
| 10092.17 | 0.93      |

If you don't know those metrics, I strongly recommend this [article](https://medium.com/analytics-vidhya/machine-learning-metrics-in-simple-terms-d58a9c85f9f6) that gives you a good overview.

----------

## **Conclusion**

If you’re an aspiring data scientist looking for a project that’s practical, impressive, and fun to build, give InsightFlow a try. It’s a great way to practice the full data science pipeline and show off your skills with the tools that matter.

**🚀 Ready to level up your portfolio?  [Check out InsightFlow on GitHub!](https://github.com/valeriova96/InsightFlow.git)**

----------

_Feel free to fork, star, and contribute. Happy coding!_

----------

**Tags:**  #DataScience #Portfolio #MachineLearning #streamlit #scikit-learn #pandas