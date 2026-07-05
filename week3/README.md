# Week 3 – Country Intelligence System using Classification, Ensemble Learning & Clustering

## Overview

This project demonstrates an end-to-end Machine Learning pipeline for analyzing and segmenting countries based on their socio-economic indicators. The workflow includes data preprocessing, exploratory data analysis (EDA), clustering, classification, ensemble learning, hyperparameter tuning, and dimensionality reduction.

> **Note:** The provided dataset contains country-level socio-economic indicators. Countries are treated as entities for segmentation and predictive modeling while implementing the required machine learning techniques.

---

## Objectives

- Perform data preprocessing and cleaning
- Conduct exploratory data analysis (EDA)
- Scale numerical features
- Segment countries using K-Means clustering
- Evaluate cluster quality using the Elbow Method and Silhouette Score
- Classify country clusters using Decision Tree, Random Forest, and XGBoost
- Optimize model performance using GridSearchCV
- Detect outliers using DBSCAN
- Visualize clusters using Principal Component Analysis (PCA)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Statsmodels

---

## Workflow

1. Data Loading & Inspection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. K-Means Clustering
6. Elbow Method & Silhouette Score
7. Cluster Analysis
8. Decision Tree Classification
9. Random Forest Classification
10. XGBoost Classification
11. Hyperparameter Tuning (GridSearchCV)
12. DBSCAN Clustering
13. PCA Visualization
14. Business Insights & Conclusion

---

## Dataset

**Country Data Dataset**

The dataset contains socio-economic indicators for countries, including:

- Country
- Child Mortality
- Exports
- Health Expenditure
- Imports
- Income
- Inflation
- Life Expectancy
- Total Fertility
- GDP per Capita

---

## Machine Learning Models

### Clustering
- K-Means
- DBSCAN

### Classification
- Decision Tree
- Random Forest
- XGBoost

### Model Optimization
- GridSearchCV

### Dimensionality Reduction
- Principal Component Analysis (PCA)

---

## Key Outcomes

- Successfully segmented countries into meaningful socio-economic groups.
- Compared multiple classification models for predicting cluster membership.
- Optimized model performance using GridSearchCV.
- Identified the most influential features using Feature Importance analysis.
- Visualized country clusters using PCA.

---

## Repository Structure

```
Week3/
│── week3_AmanNegi.ipynb
│── Country-data.csv
│── data-dictionary.csv
│── README.md
```
