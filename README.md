# ❤️ Heart Disease Prediction

This is my Heart Disease Prediction project, where I built and compared multiple machine learning models to predict whether a person is likely to have heart disease based on clinical data.
I implemented data cleaning, scaling, model training, overfitting analysis, cross-validation, and interpretability techniques to ensure high performance and explainability.

## 🖼️ Model Output

Here’s how the Overfitting Analysis visualization looks:

![Confusion Matrix](images/Screenshot%201.png)

Here’s the Feature Importance:

![Accuracy Score](images/Screenshot%202.png)

And here’s the Decision Tree Visualization:

![Accuracy Score](images/Screenshot%203.png)

Dataset[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset]

## Tools and Libraries Used

Python

Numpy

Scikit-learn – model building, evaluation, feature scaling

Matplotlib & Seaborn – data visualization

Graphviz – tree visualization export



## Project Overview

Objective: Predict whether a patient has heart disease (binary classification).

Dataset: heart.csv

Language: Python

Libraries Used: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Graphviz, 

## Steps I Implemented
### Data Preprocessing

Loaded the heart.csv dataset into a Pandas DataFrame.

Replaced missing values in the thal column (originally encoded as 0) with the mode value.

Verified data cleanliness — no missing values remaining.

Separated features (X) and target variable (y).

Performed an 80–20 train-test split with stratification to maintain class balance.

### Feature Scaling
Standardized features using StandardScaler for consistent input distribution.

### Model Training and Evaluation
#### 1. Decision Tree Classifier

Trained a baseline Decision Tree model.

Achieved accuracy: 0.985

Generated a detailed classification report.

Visualized the entire decision tree structure using plot_tree.

Performed overfitting analysis by varying tree depth (1–10).

Found the optimal max_depth = 9, balancing accuracy and generalization.

#### 2. Random Forest Classifier

Trained a Random Forest model with 100 estimators.

Achieved 100% test accuracy (perfect classification).

Performed 5-fold cross-validation, reaching average accuracy ≈ 0.997.

### Model Comparison — Confusion Matrices

I visualized both models’ confusion matrices side by side for performance comparison.

Confusion matrices were plotted using ConfusionMatrixDisplay with different color maps for better contrast.

### Feature Importance Analysis

Interpreted Random Forest feature importances to identify key predictors.

Visualized them using Seaborn barplot.

Top influential features:

cp (Chest Pain Type)

thalach (Max Heart Rate)

oldpeak (ST Depression)

ca (Number of Major Vessels)

thal (Thalassemia Type)

### Model Export

Exported the full decision tree structure as a .pdf using Graphviz.
