# Machine Learning Model Templates

A collection of reusable templates for Machine Learning (ML) models built using the `scikit-learn` library. 
This repository serves as a quick reference guide and skeleton code for common tasks such as importing models, loading datasets, performing predictions, model selection, hyperparameter tuning, and evaluating metrics.
I created this repository for myself to save time, instead of searching for examples on the internet or asking LLMs each time you work with a particular model.
If you're new to machine learning or frequently find yourself searching for information online, you can easily fork this repo for quick access to templates. However, it's highly recommended to refer to the official [scikit-learn documentation](https://scikit-learn.org/stable/) for in-depth details and best practices.


##  Contents

* **Model Templates**: Skeleton code for various ML models.
* **Hyperparameter Tuning**: Code snippets for performing hyperparameter optimization.
* **Metrics**: A set of commonly used evaluation metrics.
* **Model Selection**: Template for comparing different models.

##  Features

* **Simple Skeletons**: Easily adapt these templates for any dataset.
* **Hyperparameter Tuning**: Integrated `GridSearchCV` and `RandomizedSearchCV` examples.
* **Model Evaluation**: Includes accuracy, precision, recall, F1 score, and ROC-AUC.
* **Model Selection**: Quick framework for selecting the best model for your problem.
* **Reusable**: Works with a wide variety of machine learning tasks.


## An Example Workflow

1. **Import the Model:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
```

2. **Load Data:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3. **Train a Model:**

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

4. **Make Predictions:**

```python
y_pred = model.predict(X_test)
```

5. **Evaluate Model:**

```python
from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### Model Selection and Hyperparameter Tuning

Use **GridSearchCV** or **RandomizedSearchCV** for hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

# Define model
model = RandomForestClassifier()

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
```

### Model Comparison

You can compare different models using cross-validation scores:

```python
from sklearn.model_selection import cross_val_score

models = [RandomForestClassifier(), SVC(), LogisticRegression()]
for model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model.__class__.__name__}: {scores.mean():.4f}")
```

### Metrics

Use the following metrics to evaluate your model:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Example of calculating metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
```

##  Usage

This repository is designed to simplify and speed up your machine learning workflow. The templates provide quick access to commonly used ML techniques, enabling you to focus more on problem-solving and less on coding.
For each model, you’ll find a brief explanation of its use case, followed by the necessary code skeletons for easy integration into your own projects. Customize them for your specific dataset or problem at hand.


