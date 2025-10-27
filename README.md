# ❤️ Heart Disease Prediction

This is my Heart Disease Prediction project, where I built and compared multiple machine learning models to predict whether a person is likely to have heart disease based on clinical data.
I implemented data cleaning, scaling, model training, overfitting analysis, cross-validation, and interpretability techniques to ensure high performance and explainability.

## Tools and Libraries Used

Python

Scikit-learn – model building, evaluation, feature scaling

Matplotlib & Seaborn – data visualization

Graphviz – tree visualization export



## Project Overview

Objective: Predict whether a patient has heart disease (binary classification).

Dataset: heart.csv

Language: Python

Libraries Used: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Graphviz, Joblib

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

Model	True Negatives	False Positives	False Negatives	True Positives	Accuracy
Decision Tree	100	0	3	102	0.985
Random Forest	100	0	0	105	1.00

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
