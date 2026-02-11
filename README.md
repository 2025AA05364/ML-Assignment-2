# ML Assignment - Adult Census Income Prediction

## Problem Statement

This project implements and compares six different machine learning classification models to predict whether an individual's income exceeds $50K/year based on census data. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and deployment through an interactive Streamlit web application.

The assignment requires:
- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Adult Census Income Dataset

**Source:** [Kaggle Repository](https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset/data)

**Description:**
The dataset contains census data used to predict whether an individual's income exceeds $50K/year.

**Features (14 input features):**
1. **age**: continuous
2. **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
3. **fnlwgt**: continuous (final weight)
4. **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
5. **education-num**: continuous
6. **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
7. **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
8. **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
9. **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
10. **sex**: Female, Male
11. **capital-gain**: continuous
12. **capital-loss**: continuous
13. **hours-per-week**: continuous
14. **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

**Target Variable:**
- **income**: >50K, <=50K

**Dataset Statistics:**
- **Total Instances:** 48,842
- **Features:** 14
- **Classes:** 2 (>50K, <=50K)

## Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

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


## Installation and Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ML-Assignment - Adult Census Income Prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
    Deploy on Streamlit Community Cloud
   1. Go to https://streamlit.io/cloud
   2. Sign in using GitHub account
   3. Click “New App”
   4. Select your repository
   5. Choose branch (usually main)
   6. Select app.py
   7. Click Deploy

## Streamlit Web Application

The application includes:
- Dataset upload option (CSV) [As streamlit free tier has limited capacity,
upload only test data]
- Model selection dropdown (if multiple models)
- Display of evaluation metrics
- Confusion matrix or classification report


## Tools and Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- GitHub
- BITS Virtual Lab

## GitHub Repository

[Adult Census Income Prediction Project Repository](https://github.com/2025AA05364/ML-Assignment2-adultcensus.git)

### Streamlit App Link
[Adult Census Income Prediction App](https://adult-census-income-prediction-2025aa05364.streamlit.app/)

## References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
2. Adult Census Income dataset on Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income


## Acknowledgments
- Kaggle for the Adult Census Income dataset
- BITS Pilani for the assignment framework

## Created By
Dinesh B M