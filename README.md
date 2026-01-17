# Predictive Customer Churn Modeling for E commerce Retention Strategy

## Project Overview
End to end machine learning pipeline for predicting customer churn in e commerce, with business impact analysis and actionable insights.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-customer-churn-ecommerce-retention.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Business Problem
Despite significant user growth, e commerce platforms face critical customer churn challenges:
- Generic promotions lack personalization
- Unable to distinguish high risk customers from loyal ones
- Marketing resources wasted on unnecessary retention efforts
- Lost revenue from preventable customer churn

**Solution:** Machine learning model to identify at risk customers and enable targeted retention strategies.

## Dataset
- **Source:** HackerEarth ML Challenge 2021 - "How NOT to lose a customer in 10 days"
- **Size:** 3,941 customers × 11 features
- **Target:** Binary classification (Churn: 0/1)
- **Class Imbalance:** 82.9% Not Churned | 17.1% Churned

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Tenure | Float | Customer tenure with platform |
| WarehouseToHome | Float | Distance from warehouse to home |
| NumberOfDeviceRegistered | Integer | Total devices registered |
| PreferedOrderCat | Object | Preferred order category |
| SatisfactionScore | Integer | Satisfaction score (1-5) |
| MaritalStatus | Object | Marital status |
| NumberOfAddress | Integer | Total addresses added |
| Complain | Integer | Complaint raised (0/1) |
| DaySinceLastOrder | Float | Days since last order |
| CashbackAmount | Float | Average cashback received |
| **Churn** | Integer | **Target variable** |

## Methodology

### 1. Data Preprocessing
- **Missing Values:** KNN Imputer (n_neighbors=10)
- **Transformation:** Power Transformer (Yeo-Johnson)
- **Scaling:** Robust Scaler (outlier-resistant)
- **Encoding:** One-Hot Encoding for categorical features
- **Cleaning:** Removed 671 duplicates, merged "Mobile" categories

### 2. Class Imbalance Handling
Tested 6 sampling techniques - **RandomOverSampler** selected for best F2 performance:
- RandomOverSampler (Selected)
- SMOTE
- BorderlineSMOTE
- ClusterCentroids
- TomekLinks
- NearMiss

### 3. Model Selection & Evaluation
Evaluated 8 algorithms with 5 fold Stratified Cross Validation:

| Model | F2 Score | 
|-------|----------|
| **LightGBM** | **0.892** | 
| XGBoost | 0.886 | 
| Logistic Regression | 0.822 | 
| Decision Tree | 0.665 | 
| Ada Boost | 0.663 | 
| Gradient Boosting | 0.611 | 
| Random Forest | 0.591 | 
| KNN | 0.466 | 

### 4. Hyperparameter Tuning
**RandomizedSearchCV** with F2 score optimization (β=2):

```python
Best Parameters:
{
    'subsample': 0.8,
    'reg_lambda': 0.5,
    'reg_alpha': 0.5,
    'num_leaves': 50,
    'n_estimators': 100,
    'min_child_samples': 20,
    'max_depth': 20,
    'learning_rate': 0.05,
    'colsample_bytree': 0.6
}

``` 

## Model Performance

### Final Metrics (Test Set)
| Metric | Score |
|--------|-------|
| **F2 Score** | **0.841** |
| **Recall** | **88.8%** |
| **Precision** | 69.3% |
| **Accuracy** | 91.7% |
| **ROC-AUC** | 0.964 |
| **PR-AUC** | 0.808 |

### Confusion Matrix
|  | Predicted Not Churn | Predicted Churn |
|---|---------------------|-----------------|
| **Actual Not Churn** | 505 (TN) | 42 (FP) |
| **Actual Churn** | 12 (FN) | 95 (TP) |

### Why F2 Score?
- Prioritizes **Recall** over Precision
- Missing a churner costs more than unnecessary retention offer
- Enables early intervention for at-risk customers
- Aligns with business goal: maximize customer retention

## Business Impact Analysis

### ROI Calculation
**Assumptions:**
- Customer Lifetime Value: Rs. 350,000
- Retention Offer Cost: Rs. 20,000
- Retention Success Rate: 70%

**Results:**
| Metric | Value |
|--------|-------|
| Identified At-Risk Customers | 137 |
| Investment (Retention Offers) | Rs. 6,850 |
| Successfully Retained (70%) | 95 customers |
| Revenue Saved | Rs. 115,079 |
| **Net Profit** | **Rs. 108,229** |
| **ROI** | **1,580%** |

## Key Insights

### Top 5 Churn Drivers (SHAP Analysis)
1. **Tenure** (1.33) - Strongest predictor; <2 months = 5× higher risk
2. **Complain** (0.65) - 63% of churners had unresolved complaints
3. **CashbackAmount** (0.40) - Churners receive 10% less cashback
4. **NumberOfAddress** (0.33) - Multiple addresses indicate relocation
5. **WarehouseToHome** (0.28) - >20km distance = 40% higher risk

### Churner Profile
| Characteristic | Churned | Non-Churned | Difference |
|----------------|---------|-------------|------------|
| Avg. Tenure | 2.4 months | 10.8 months | **-77%** |
| Complaint Rate | 61.7% | 24.1% | **+156%** |
| Avg. Cashback | Rs. 158 | Rs. 177 | **-10%** |
| Avg. Distance | 17.1 km | 15.2 km | **+13%** |
| Days Since Order | 3.0 days | 4.7 days | **-36%** |

## Business Recommendations

### 1. Early Tenure Customer Retention
- Implement 60-day onboarding programs
- Offer targeted promotions for customers <2 months
- Proactive check-ins at 30 and 45 days
- Celebrate early wins with rewards

### 2. Enhance Complaint Resolution
- 24-hour SLA for initial response, 72-hour resolution
- Proactive sentiment analysis on communications
- Standardized compensation framework
- Closed-loop feedback post-resolution

### 3. Optimize Cashback & Loyalty
- Personalized cashback tiers based on behavior
- "Win-back" offers for high-risk customers
- Reward non-purchase behaviors (reviews, referrals)
- Improve cashback visibility in dashboard

### 4. Improve Delivery Experience
- Partner with regional logistics for localized fulfillment
- Real-time tracking with proactive delay notifications
- Offer flat-rate/free shipping subscriptions
- Flexible delivery windows and locations

### 5. Targeted Retention Campaigns
- **High Risk (>0.7):** Personal outreach + high-value offers
- **Medium Risk (0.4-0.7):** Automated campaigns + moderate incentives
- **Low Risk (<0.4):** General loyalty engagement

### 6. Implementation Timeline
- **Immediate (0-30 days):** Deploy scoring, high-risk outreach
- **Short-term (1-3 months):** Automated campaigns, optimize cashback
- **Medium-term (3-6 months):** Onboarding program, delivery enhancement
- **Long-term (6-12 months):** Product improvements, advanced personalization

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/nayana-sisil/predictive-customer-churn-ecommerce-retention.git
cd predictive-customer-churn-ecommerce-retention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

App will open at `http://localhost:8501`

## Usage

### Web Application
1. Visit [Live Demo](https://predictive-customer-churn-ecommerce-retention.streamlit.app/)
2. Enter customer information
3. Click "Predict Churn"
4. View probability, risk classification, and insights

### Python API
```python
import pickle
import pandas as pd

# Load model
with open('churn_model_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare data
customer_data = pd.DataFrame({
    'Tenure': [2.0],
    'WarehouseToHome': [18.0],
    'NumberOfDeviceRegistered': [4],
    'PreferedOrderCat': ['Mobile Phone'],
    'SatisfactionScore': [3],
    'MaritalStatus': ['Single'],
    'NumberOfAddress': [5],
    'Complain': [1],
    'DaySinceLastOrder': [3.0],
    'CashbackAmount': [150.0]
})

# Predict
prediction = model.predict(customer_data)
probability = model.predict_proba(customer_data)[:, 1]

print(f"Churn: {prediction[0]}")
print(f"Probability: {probability[0]:.2%}")
```

## Project Structure
```
predictive-customer-churn-ecommerce-retention/
├── Dataset/
│   └── data_ecommerce_customer_churn.csv
├── Final_report_and_analysis/
│   └── Final_report.ipynb
├── Reports/
│   ├── images  
├── app.py                          # Streamlit application
├── churn_model_final.pkl          # Trained model
├── requirements.txt               # Python dependencies
├── packages.txt                   # System packages
├── runtime.txt                    # Python version
├── .gitignore
└── README.md
```

## Technologies Used

### Core ML Stack
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms & preprocessing
- **LightGBM** - Gradient boosting model
- **Imbalanced-learn** - Sampling techniques

### Visualization & Explainability
- **Matplotlib** - Static plots
- **Seaborn** - Statistical graphics
- **Plotly** - Interactive charts
- **SHAP** - Model interpretability

### Deployment
- **Streamlit** - Web application
- **Pickle** - Model serialization

## Future Enhancements
- [ ] Real-time prediction API (FastAPI)
- [ ] Multi-model ensemble approach
- [ ] Customer segmentation clustering
- [ ] Time-series churn forecasting
- [ ] A/B testing framework
- [ ] Deep learning models (LSTM/Transformers)
- [ ] Automated retraining pipeline
- [ ] Sentiment analysis integration

## Contributing
Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Contact
**Nayana Sisil**

- 📧 Email: nayanasisil@gmail.com
- 📱 WhatsApp: +94 76 860 9939
- 🔗 GitHub: [@nayana-sisil](https://github.com/nayana-sisil)
- 🌐 Live Demo: [Streamlit App](https://predictive-customer-churn-ecommerce-retention.streamlit.app/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Dataset:** HackerEarth ML Challenge 2021
- **Reference:** Zhang, X., Ghosh, A., & Ali, A. (2024). Research on Customer Retention Strategy in the E-commerce Environment. *Economics & Management Review, 5, 8*.
- **Inspiration:** Data-driven retention strategies for competitive e-commerce markets

---

 **If this project helps you, please give it a star!** 

**Last Updated:** January 2026
```
