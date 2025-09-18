# Customer Churn Analysis and Prediction  

![Analysis Preview](quarto/capa.png)  

### ➡️ Full and Detailed Analysis in My Portfolio:  
[Access the full project page here](https://ferreiragabrielw.github.io/portfolio-gabriel/projetos/DataScience/2CustomerChurn/CustomerChurn.html)  

### 🚀 Deployed:  
[Try the live app here](https://ds-churn-prediction-ml-deploy-qp3znuqfw4fpwqcv8li3n2.streamlit.app/)  

---

## About the Project  

This Data Science project focuses on predicting customer churn in a fictional telecommunications company. The dataset contains information from 7,043 customers in California during the third quarter, including demographics, services used, account information, billing, and whether the customer canceled their contract.  

The main objective is twofold:  
1. Predict churn (binary classification) using Machine Learning models.  
2. Interpret churn drivers to identify the main reasons behind customer attrition, enabling the company to develop proactive retention strategies.  

---

## Technologies and Process  

* Tools: Python (Pandas, Scikit-learn, LightGBM, XGBoost, Matplotlib, Seaborn), Quarto, Jupyter Notebook, Streamlit.  
* End-to-End Analysis Pipeline (E2E):  
    * Data Cleaning & Preparation:  
        - Handled missing values in `TotalCharges` (imputed with 0 for new customers).  
        - Standardized categories (`No internet/phone service` → `No`).  
        - Dropped irrelevant column (`customerID`).  
    * Exploratory Data Analysis (EDA):  
        - Churn rate: ~27% of customers left.  
        - High early churn within the first 9 months of tenure.  
        - Higher churn among customers with month-to-month contracts, paperless billing, and electronic check payments.  
    * Feature Engineering & Encoding:  
        - Label Encoding for binary variables.  
        - One-Hot Encoding for categorical features.  
        - Removed `TotalCharges` due to strong multicollinearity with `tenure` and `MonthlyCharges`.  
    * Class Imbalance Handling:  
        - Tested class_weight balancing and SMOTETomek resampling.  
    * Modeling & Evaluation:  
        - Tested multiple ML algorithms: Logistic Regression, Random Forest, Decision Tree, SVM, KNN, XGBoost, LightGBM.  
        - Evaluated with Recall as the main metric (to maximize detection of churners), plus Precision, F1-Score, and AUC-ROC.  
    * Hyperparameter Optimization:  
        - GridSearchCV for Logistic Regression, Random Forest, SVM, XGBoost, LightGBM.  
        - Tuned models consistently improved Recall and AUC-ROC.  

* Key Insights (Business Perspective):  
    * High churn risk is associated with:  
        - Short tenure (customers within the first year).  
        - Month-to-month contracts.  
        - Electronic check payments.  
        - Lack of value-added services (tech support, online security).  
    * Strategic Recommendations:  
        - Strengthen onboarding in the first 6–9 months.  
        - Incentivize longer-term contracts with benefits/discounts.  
        - Address billing dissatisfaction (especially among high-paying customers).  
        - Bundle services to increase stickiness.  

---

## Repository Content  

* `data/`: Contains the raw dataset (`Telco-Customer-Churn.csv`).  
* `notebooks/`: Includes the full analysis Jupyter Notebook (`.ipynb`) with all steps (EDA, preprocessing, modeling, interpretation).  
* `quarto/`: Includes the `.qmd` source file of the report and its rendered HTML version.  
* `README.md`: This document.  
* `LICENSE`: Project license (MIT License).  

---

## How to View the Full Analysis  

* **Online (HTML)**: Download the `CustomerChurn.html` file from the `quarto/` folder and open it in your browser.  
* **Jupyter Notebook**: Explore the `.ipynb` in the `notebooks/` folder directly on GitHub or locally.  
* **Locally (Quarto)**:  
    1. Download the `CustomerChurn.qmd` file from the `quarto/` folder.  
    2. Install Quarto and Python with the required libraries.  
    3. Open the `.qmd` file in VS Code (with Quarto extension) and render it.  

---

### License  

This project is licensed under the [MIT License](LICENSE).  
