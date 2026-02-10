**1. Problem Statement**

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard evaluation metrics, and deploy the trained models through an interactive Streamlit web application. The assignment demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

**2. Dataset Description**

The Wine Quality (Red) dataset was used for this assignment. The dataset is publicly available from the UCI Machine Learning Repository and is also hosted on Kaggle for convenience.

The dataset contains physicochemical properties of Portuguese “Vinho Verde” red wine samples. Due to privacy and logistical reasons, only physicochemical input variables and sensory output variables are provided.

**Number of instances**: 1,599

**Number of features:** 11 input features

**Target variable**: Wine quality score (originally ranging from 0 to 10)

Input Features:
    Fixed acidity
    Volatile acidity
    Citric acid
    Residual sugar
    Chlorides
    Free sulfur dioxide
    Total sulfur dioxide
    Density
    pH
    Sulphates
    Alcohol
    Target Variable Transformation
For this assignment, the problem was formulated as a binary classification task, following the recommendation in the dataset documentation:

Wines with quality ≥ 6 were labeled as Good (1)
Wines with quality < 6 were labeled as Bad (0)

This transformation enables effective use of classification metrics such as AUC and MCC.

**3. Models Used and Evaluation Metrics**
The following six classification models were implemented using the same dataset and train–test split:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes Classifier
Random Forest (Ensemble Model)
XGBoost (Ensemble Model)

Each model was evaluated using the following mandatory metrics:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

**Model Comparison Table**

	Model	            Accuracy	AUC     	Precision	Recall	    F1 Score	MCC
0	Logistic Regression	0.740625	0.824208	0.768293	0.736842	0.752239	0.480819
1	Decision Tree	    0.756250	0.754661	0.768786	0.777778	0.773256	0.509802
2	KNN	                0.734375	0.805251	0.752941	0.748538	0.750733	0.466467
3	Naive Bayes	        0.721875	0.788375	0.773333	0.678363	0.722741	0.449989
4	Random Forest	    0.806250	0.902665	0.830303	0.801170	0.815476	0.612098
5	XGBoost	            0.825000	0.896346	0.848485	0.818713	0.833333	0.649705

**4. Observations on Model Performance**
ML Model	Observation
Logistic Regression	Performs reasonably well after feature scaling and provides a strong baseline model.
Decision Tree	Captures non-linear relationships but shows tendency to overfit the training data.
KNN	Performance depends on distance calculation and benefits significantly from feature scaling.
Naive Bayes	Fast and efficient but assumes feature independence, which may limit performance.
Random Forest	Provides better generalization and robustness by combining multiple decision trees.
XGBoost	Achieves the best overall performance due to gradient boosting and effective handling of feature interactions.

**5. Streamlit Web Application**

An interactive Streamlit web application was developed and deployed using Streamlit Community Cloud. The application includes:
CSV file upload option for test data
Model selection dropdown
Display of evaluation metrics
Confusion matrix / classification report for predictions

**6. Tools and Technologies Used**

Python
Pandas, NumPy
Scikit-learn
XGBoost
Streamlit
GitHub
BITS Virtual Lab

**7. Reference**

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis,
Modeling wine preferences by data mining from physicochemical properties,
Decision Support Systems, Elsevier, 47(4):547–553, 2009.