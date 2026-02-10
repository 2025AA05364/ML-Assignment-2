# ML Assignment 3: Adult Census Income Prediction

## 1. Problem Statement

The objective of this project is to implement multiple machine learning classification models to predict whether a person makes over $50K a year based on census data. We evaluate the models' performance using standard evaluation metrics and deploy the trained models through an interactive Streamlit web application. This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

## 2. Dataset Description

The Adult Census Income dataset is used for this assignment. This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The dataset is available on Kaggle: [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income).

Dataset characteristics:
- Extraction was done by Barry Becker from the 1994 Census database.
- A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)).
- Prediction task is to determine whether a person makes over 50K a year.
- Number of instances: 48,842
- Number of attributes: 14 (mixed categorical and numerical)

Attribute Information:
1. age: continuous
2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
3. fnlwgt: continuous (final weight)
4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
5. education-num: continuous
6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
10. sex: Female, Male
11. capital-gain: continuous
12. capital-loss: continuous
13. hours-per-week: continuous
14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

Target Variable:
- income: >50K, <=50K

## 3. Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| Model                | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.830598 | 0.864159| 0.755319  | 0.472703| 0.581491 | 0.503076|
| Decision Tree        | 0.810874 | 0.746798| 0.620414  | 0.619174| 0.619793 | 0.493925|
| KNN                  | 0.834411 | 0.862154| 0.686989  | 0.615180| 0.649104 | 0.542585|
| Naive Bayes          | 0.797779 | 0.861328| 0.685039  | 0.347537| 0.461131 | 0.383437|
| Random Forest        | 0.864412 | 0.913774| 0.774038  | 0.643142| 0.702545 | 0.620138|
| XGBoost              | 0.874192 | 0.931772| 0.790008  | 0.673768| 0.727273 | 0.649636|

### Observations on Model Performance

| ML Model            | Observation                                                                                                    |
|---------------------|----------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Performs well with high accuracy and AUC, but lower recall compared to ensemble methods.                       |
| Decision Tree       | Shows balanced precision and recall, but overall performance is lower than other models.                       |
| KNN                 | Similar performance to Logistic Regression, with good accuracy and AUC but lower recall.                       |
| Naive Bayes         | Lowest overall performance, particularly struggling with recall and F1 score.                                  |
| Random Forest       | Strong performance across all metrics, second only to XGBoost.                                                 |
| XGBoost             | Best overall performance with highest accuracy, AUC, and MCC scores.                                           |

These results demonstrate the superiority of ensemble methods (XGBoost and Random Forest) for this particular dataset and classification task.

## 4. Streamlit Web Application

An interactive Streamlit web application was developed and deployed using Streamlit Community Cloud. The application includes:
- CSV file upload option for test data
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix / classification report for predictions

### Streamlit App Link
[Adult Census Income Prediction App](https://ml-assignment-2-2025aa05364.streamlit.app/)

## 5. Tools and Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- GitHub
- BITS Virtual Lab

## 6. GitHub Repository

[Adult Census Income Prediction Project Repository](https://github.com/2025AA05364/ML-Assignment-3.git)

## 7. References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

2. Adult Census Income dataset on Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income