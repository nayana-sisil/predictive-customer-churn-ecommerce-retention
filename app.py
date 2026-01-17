import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc,
    fbeta_score, accuracy_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="E-Commerce Customer Churn Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        background: linear-gradient(90deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
    }
    
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-left: 10px;
        border-left: 4px solid #3498db;
    }
    
    .metric-card {
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .highlight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .highlight-card h3,
    .highlight-card p {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        padding: 0;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        margin-right: 2px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
        font-weight: 600;
    }
    
    .plot-container {
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-bottom: 1.5rem;
    }
    
    .savings-positive {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: rgba(39, 174, 96, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    .savings-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: rgba(231, 76, 60, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    [data-testid="stNumberInput"],
    [data-testid="stSlider"],
    [data-testid="stRadio"],
    [data-testid="stPlotlyChart"],
    .stForm {
        background-color: transparent !important;
    }
    
    [data-testid="stNumberInput"] > div,
    [data-testid="stSlider"] > div,
    [data-testid="stRadio"] > div,
    [data-testid="stPlotlyChart"] > div {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stSlider"] > div > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .dataframe th {
        background-color: #3498db !important;
        color: white !important;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
    }
    
    .dataframe td {
        padding: 10px 15px;
    }
    
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    .spacing {
        margin-top: 2rem;
    }
    
    .markdown-text {
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .markdown-text ul, .markdown-text ol {
        padding-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .markdown-text li {
        margin-bottom: 0.5rem;
    }
    
    [data-baseweb="input"],
    [data-baseweb="slider"],
    [data-baseweb="radio"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_model():
    import os
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'churn_model_final.pkl')
        
        # Debug: Show what files exist
        st.write(f"Looking for model at: {model_path}")
        st.write(f"Current directory: {current_dir}")
        st.write(f"Files in directory: {os.listdir(current_dir)}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model file not found. Error: {str(e)}")
        st.info("Please ensure 'churn_model_final.pkl' is in the same directory as app.py")
        return None

@st.cache_data
def load_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Tenure': np.random.uniform(0, 30, n_samples),
        'WarehouseToHome': np.random.uniform(5, 40, n_samples),
        'NumberOfDeviceRegistered': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'PreferedOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], n_samples),
        'SatisfactionScore': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'NumberOfAddress': np.random.randint(1, 10, n_samples),
        'Complain': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'DaySinceLastOrder': np.random.uniform(0, 15, n_samples),
        'CashbackAmount': np.random.uniform(100, 350, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    churn_prob = (
        (df['Tenure'] < 5) * 0.4 +
        (df['Complain'] == 1) * 0.3 +
        (df['SatisfactionScore'] < 3) * 0.2 +
        (df['DaySinceLastOrder'] > 7) * 0.1
    )
    df['Churn'] = np.random.binomial(1, churn_prob.clip(0, 1))
    
    return df

def get_plot_template():
    return 'plotly_white'

model = load_model()
df = load_sample_data()

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='margin: 0; font-size: 1.5rem; color: white;'>E Commerce Analytics</h3>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9; color: white;'>Customer Churn Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select Dashboard Section",
        ["Business Overview", "Model Analysis", "Churn Prediction", "Financial Assessment"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### Business Parameters")
    customer_lifetime_value = st.number_input(
        "Customer Lifetime Value (Rs.)",
        min_value=10000,
        max_value=1000000,
        value=350000,
        step=50000,
        help="Estimated revenue from a customer over their lifetime"
    )
    
    retention_offer_cost = st.number_input(
        "Retention Offer Cost (Rs.)",
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help="Cost of retention offer per customer"
    )
    
    retention_success_rate = st.slider(
        "Retention Success Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Percentage of customers who stay after receiving retention offer"
    ) / 100
    
    st.markdown("---")
    
    st.markdown("### Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Churn Rate", f"{(df['Churn'].mean()*100):.1f}%")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; opacity: 0.7; font-size: 0.8rem;'>
        Built for Customer Experience Optimization
    </div>
    """, unsafe_allow_html=True)

#Main content area
if page == "Business Overview":
    st.markdown('<div class="main-header">E Commerce Customer Churn Analytics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Churn Rate",
            f"{df['Churn'].mean()*100:.1f}%",
            delta=f"-{5.2:.1f}% target",
            delta_color="inverse"
        )
    
    with col2:
        revenue_at_risk = df['Churn'].sum() * customer_lifetime_value
        st.metric(
            "Revenue at Risk",
            f"Rs.{revenue_at_risk:,.0f}",
            delta=f"-Rs.{(revenue_at_risk * 0.3):,.0f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Model Coverage",
            "88.9%",
            "Recall Score",
            delta_color="normal"
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Challenge")
        st.write("The e commerce platform has experienced significant user growth but faces increasing customer churn. Current retention strategies lack personalization and data driven targeting, reducing their effectiveness.")
        
        st.markdown("### Solution")
        st.markdown("""
        - Targeted retention campaigns
        - Personalized offers and recommendations
        - Optimized marketing resource allocation
        - Proactive customer engagement
        """)
        
        st.markdown("### Business Impact")
        st.markdown("""
        1. Improve customer loyalty and lifetime value
        2. Reduce marketing spend waste
        3. Increase retention campaign effectiveness
        4. Enhance overall customer experience
        """)
    
    with col2:
        #quick churn distribution
        fig = go.Figure(data=[
            go.Pie(
                labels=['Retained', 'Churned'],
                values=[df['Churn'].value_counts()[0], df['Churn'].value_counts()[1]],
                hole=.3,
                marker=dict(colors=['#27ae60', '#e74c3c']),
                textinfo='label+percent',
                textposition='inside'
            )
        ])
        fig.update_layout(
            title='Customer Distribution',
            height=300,
            showlegend=False,
            margin=dict(t=50, b=0, l=0, r=0),
            template=get_plot_template()
        )
        st.plotly_chart(fig, use_container_width=True)
        
    
    st.markdown('<div class="section-header">Key Metrics Distribution</div>', unsafe_allow_html=True)
    
    #feature distributions
    features_to_plot = ['Tenure', 'CashbackAmount', 'DaySinceLastOrder', 'SatisfactionScore']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=features_to_plot,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, feature in enumerate(features_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        for churn_value, color in zip([0, 1], colors):
            subset = df[df['Churn'] == churn_value]
            fig.add_trace(
                go.Histogram(
                    x=subset[feature],
                    name='Churned' if churn_value == 1 else 'Retained',
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        bargap=0.1,
        template=get_plot_template()
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Analysis":
    st.markdown('<div class="main-header">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        " Performance Metrics", 
        " Feature Importance", 
        " Confusion Analysis", 
        " Model Comparison"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Model Performance Summary</div>', unsafe_allow_html=True)
        
        #simulated model performance metrics
        performance_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'ROC-AUC'],
            'Value': [0.917, 0.693, 0.889, 0.779, 0.841, 0.964],
            'Description': [
                'Overall correct predictions',
                'True churn predictions among all churn predictions',
                'True churn predictions among actual churners',
                'Harmonic mean of precision and recall',
                'Weighted towards recall (β=2)',
                'Area under ROC curve'
            ]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Numerical Metrics")
            st.dataframe(
                perf_df.style.format({'Value': '{:.3f}'}),
                use_container_width=True
            )
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=perf_df['Metric'],
                    y=perf_df['Value'],
                    text=perf_df['Value'].apply(lambda x: f'{x:.3f}'),
                    textposition='auto',
                    marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
                )
            ])
            
            fig.update_layout(
                title='Model Performance Metrics',
                yaxis_title='Score',
                yaxis_range=[0, 1],
                height=400,
                template=get_plot_template()
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="subsection-header">F2-Score Justification</div>', unsafe_allow_html=True)
        st.info("""
        The F2-score (β=2) is optimized as the primary metric because:
        
        1. **Business Priority**: Missing a churning customer (false negative) costs significantly more than an unnecessary retention offer (false positive)
        2. **Recall Focus**: Gives twice the weight to recall compared to precision
        3. **Cost Sensitivity**: Aligns with business goal of aggressively reducing churn
        4. **Resource Optimization**: Enables effective intervention on high risk customers
        """)
    
    with tab2:
        st.markdown('<div class="subsection-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        #feature importance data
        feature_importance = {
            'Feature': [
                'CashbackAmount', 'WarehouseToHome', 'Tenure', 'DaySinceLastOrder',
                'NumberOfAddress', 'SatisfactionScore', 'NumberOfDeviceRegistered',
                'Complain', 'MaritalStatus_Single', 'MaritalStatus_Married'
            ],
            'Importance': [1240, 911, 663, 585, 377, 346, 261, 116, 111, 68]
        }
        
        fi_df = pd.DataFrame(feature_importance)
        fi_df = fi_df.sort_values('Importance', ascending=True)
        
        #create feature importance plot
        fig = go.Figure(data=[
            go.Bar(
                y=fi_df['Feature'],
                x=fi_df['Importance'],
                orientation='h',
                marker_color='#3498db',
                text=fi_df['Importance'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Top 10 Most Important Features',
            xaxis_title='Importance Score',
            height=500,
            template=get_plot_template()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Insights")
            st.markdown("""
            - **CashbackAmount**: Most influential feature – incentives strongly affect retention  
            - **WarehouseToHome**: Delivery distance impacts customer satisfaction  
            - **Tenure**: Longer-tenured customers are more loyal  
            - **Recent Activity**: *DaySinceLastOrder* is critical for churn prediction
            """)
        
        with col2:
            st.markdown("### Business Implications")
            st.markdown("""
            1. **Enhance Loyalty Programs**: Focus on cashback and rewards
            2. **Optimize Logistics**: Reduce delivery distance impact
            3. **Early Intervention**: Target new customers with onboarding
            4. **Proactive Engagement**: Monitor recent purchase patterns
            """)
    
    with tab3:
        st.markdown('<div class="subsection-header">Confusion Matrix Analysis</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            #confusion matrix
            cm = np.array([[503, 44], [12, 95]])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Not Churn', 'Predicted Churn'],
                y=['Actual Not Churn', 'Actual Churn'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix - Final Model',
                height=400,
                width=500,
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template=get_plot_template()
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown('<div class="subsection-header">Performance Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Positives", tp, "Correct churn predictions")
        
        with col2:
            st.metric("False Positives", fp, "Incorrect churn predictions")
        
        with col3:
            st.metric("False Negatives", fn, "Missed churners")
        
        with col4:
            st.metric("True Negatives", tn, "Correct non churn predictions")
        
        st.markdown('<div class="subsection-header">Error Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **False Positives (Cost: Rs.{fp * retention_offer_cost:,})**
            
            Customers predicted to churn but actually stay.
            
            **Business Impact**: 
            - Unnecessary retention offers
            - Wasted marketing budget
            - No customer loss
            
            **Acceptable trade off** for higher recall
            """)
        
        with col2:
            st.error(f"""
            **False Negatives (Cost: Rs.{fn * customer_lifetime_value:,})**
            
            Customers predicted to stay but actually churn.
            
            **Business Impact**:
            - Lost customers
            - Lost future revenue
            - Failed retention opportunity
            
            **Minimized by F2 score optimization**
            """)
    
    with tab4:
        st.markdown('<div class="subsection-header">Model Comparison</div>', unsafe_allow_html=True)
        
        models_comparison = {
            'Model': ['LightGBM + RandomOverSampler', 'XGBoost + RandomOverSampler', 
                     'Logistic Regression + NearMiss', 'Random Forest + Class Weight',
                     'Decision Tree + Class Weight'],
            'F2-Score': [0.892, 0.886, 0.822, 0.592, 0.665],
            'Recall': [0.889, 0.856, 0.822, 0.587, 0.665],
            'Precision': [0.693, 0.742, 0.701, 0.591, 0.664],
            'Training Time (s)': [8.2, 12.5, 3.1, 15.8, 2.3]
        }
        
        comp_df = pd.DataFrame(models_comparison)
        
        st.dataframe(
            comp_df.style.format({
                'F2-Score': '{:.3f}', 
                'Recall': '{:.3f}', 
                'Precision': '{:.3f}', 
                'Training Time (s)': '{:.1f}'
            }),
            use_container_width=True
        )
        
        #visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comp_df['Recall'],
            y=comp_df['Precision'],
            mode='markers+text',
            marker=dict(
                size=comp_df['F2-Score'] * 50,
                color=comp_df['F2-Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='F2-Score')
            ),
            text=comp_df['Model'],
            textposition='top center',
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title='Model Comparison: Precision vs Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            template=get_plot_template()
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Churn Prediction":
    st.markdown('<div class="main-header">Customer Churn Prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="subsection-header">Customer Profile Input</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                tenure = st.slider("Tenure (months)", 0.0, 30.0, 10.0, 0.5)
                warehouse_to_home = st.slider("Warehouse to Home Distance (km)", 5.0, 40.0, 15.0, 0.5)
                num_devices = st.selectbox("Number of Devices Registered", [1, 2, 3, 4, 5, 6])
                preferred_category = st.selectbox(
                    "Preferred Order Category",
                    ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others']
                )
            
            with col1_2:
                satisfaction_score = st.select_slider("Satisfaction Score", options=[1, 2, 3, 4, 5], value=3)
                marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
                num_addresses = st.slider("Number of Addresses", 1, 15, 3, 1)
                has_complaint = st.selectbox("Recent Complaint", ['No', 'Yes'])
                days_since_last = st.slider("Days Since Last Order", 0.0, 30.0, 7.0, 0.5)
                cashback_amount = st.slider("Cashback Amount (Rs.)", 100.0, 500.0, 200.0, 10.0)
            
            predict_button = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Quick Insights</div>', unsafe_allow_html=True)
        
        risk_factors = []
        
        if tenure < 5:
            risk_factors.append("Low tenure (< 5 months)")
        if has_complaint == 'Yes':
            risk_factors.append("Recent complaint")
        if satisfaction_score < 3:
            risk_factors.append("Low satisfaction score")
        if days_since_last > 7:
            risk_factors.append("Inactive for > 7 days")
        if cashback_amount < 150:
            risk_factors.append("Low cashback amount")
        
        if risk_factors:
            st.warning("#### Risk Factors Detected")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("#### No Major Risk Factors")
            st.markdown("Customer profile shows good retention indicators")
        
        st.markdown("---")
        st.markdown("#### Top Churn Indicators")
        st.markdown("""
        1. Low cashback amount
        2. Recent complaints
        3. Short customer tenure
        4. High warehouse distance
        5. Recent inactivity
        """)
    
    if predict_button:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        input_data = {
            'Tenure': tenure,
            'WarehouseToHome': warehouse_to_home,
            'NumberOfDeviceRegistered': num_devices,
            'PreferedOrderCat': preferred_category,
            'SatisfactionScore': satisfaction_score,
            'MaritalStatus': marital_status,
            'NumberOfAddress': num_addresses,
            'Complain': 1 if has_complaint == 'Yes' else 0,
            'DaySinceLastOrder': days_since_last,
            'CashbackAmount': cashback_amount
        }
        
        #convert to DataFrame for display
        input_df = pd.DataFrame([input_data])
        
        #display input summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Customer Profile Summary")
            styled_df = input_df.T.rename(columns={0: 'Value'})
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            #simulate prediction
            churn_probability = (
                (tenure < 5) * 0.3 +
                (has_complaint == 'Yes') * 0.25 +
                (satisfaction_score < 3) * 0.2 +
                (days_since_last > 7) * 0.15 +
                (cashback_amount < 150) * 0.1
            )
            
            churn_probability = min(churn_probability, 0.95)
            churn_prediction = churn_probability > 0.5
            
            # Display prediction
            if churn_prediction:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: rgba(231, 76, 60, 0.15); 
                            border-radius: 10px; border: 2px solid #dc3545; margin-top: 20px;'>
                    <h1 style='margin: 0;'>⚠️</h1>
                    <h2 style='color: #dc3545; margin: 1rem 0;'>HIGH CHURN RISK</h2>
                    <p style='font-size: 1.2rem;'>Probability: {churn_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: rgba(39, 174, 96, 0.15); 
                            border-radius: 10px; border: 2px solid #28a745; margin-top: 20px;'>
                    <h1 style='margin: 0;'>✅</h1>
                    <h2 style='color: #28a745; margin: 1rem 0;'>LOW CHURN RISK</h2>
                    <p style='font-size: 1.2rem;'>Probability: {churn_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        #risk gauge
        st.markdown("#### Risk Assessment")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score", 'font': {'size': 24}},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#e74c3c" if churn_prediction else "#27ae60"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(39, 174, 96, 0.2)"},
                    {'range': [30, 70], 'color': "rgba(255, 193, 7, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template=get_plot_template()
        )
        st.plotly_chart(fig, use_container_width=True)
        
        #recommendations
        st.markdown("#### Retention Recommendations")
        
        if churn_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Immediate Actions (Within 24 hours):**
                
                - Personal outreach from customer success
                - Special retention offer (20% discount)
                - Priority support assignment
                - Follow-up call scheduled
                
                **Expected Cost:** Rs.{retention_offer_cost:,}
                """)
            
            with col2:
                st.info(f"""
                **Strategic Actions (Next 7 days):**
                
                - Review complaint history
                - Analyze purchase patterns
                - Customize future offers
                - Monitor engagement metrics
                
                **Potential Savings:** Rs.{customer_lifetime_value:,}
                """)
        else:
            st.info("""
            **Maintenance Actions:**
            
            - Continue standard engagement
            - Monitor satisfaction scores
            - Regular promotional offers
            - Quarterly check-ins
            
            **No immediate intervention required.** Focus resources on higher risk customers.
            """)

elif page == "Financial Assessment":
    st.markdown('<div class="main-header">Financial Impact Assessment</div>', unsafe_allow_html=True)
    
    #simulation controls
    st.markdown('<div class="subsection-header">Simulation Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_customers = st.number_input(
            "Total Customer Base",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col2:
        baseline_churn_rate = st.slider(
            "Baseline Churn Rate (%)",
            min_value=1.0,
            max_value=30.0,
            value=17.1,
            step=0.1
        ) / 100
    
    with col3:
        model_precision = st.slider(
            "Model Precision (%)",
            min_value=50.0,
            max_value=95.0,
            value=69.3,
            step=0.1
        ) / 100
    
    st.markdown('<div class="subsection-header">Financial Simulation</div>', unsafe_allow_html=True)
    
    #calculate financial impact
    actual_churners = int(num_customers * baseline_churn_rate)
    
    model_recall = 0.889  
    model_fpr = 0.08  #estimated false positive rate
    
    #calculate predictions
    true_positives = int(actual_churners * model_recall)
    false_negatives = actual_churners - true_positives
    false_positives = int((num_customers - actual_churners) * model_fpr)
    
    #cost calculations
    baseline_cost = actual_churners * customer_lifetime_value
    
    intervention_cost = (true_positives + false_positives) * retention_offer_cost
    lost_revenue = false_negatives * customer_lifetime_value
    saved_revenue = true_positives * retention_success_rate * customer_lifetime_value
    model_cost = intervention_cost + lost_revenue
    
    net_savings = baseline_cost - model_cost
    roi = (net_savings / intervention_cost) * 100 if intervention_cost > 0 else 0
    
    #display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Revenue at Risk",
            f"Rs.{baseline_cost:,.0f}",
            delta=f"{actual_churners} customers",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Intervention Cost",
            f"Rs.{intervention_cost:,.0f}",
            delta=f"{true_positives + false_positives} offers",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Revenue Saved",
            f"Rs.{saved_revenue:,.0f}",
            delta=f"{int(true_positives * retention_success_rate)} retained",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Net Savings",
            f"Rs.{net_savings:,.0f}",
            delta=f"ROI: {roi:.0f}%",
            delta_color="inverse" if net_savings < 0 else "normal"
        )
    
    #visualization
    st.markdown('<div class="subsection-header">Cost Benefit Analysis</div>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cost Breakdown', 'Financial Impact'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    #pie chart 
    cost_labels = ['Retention Offers', 'Lost Revenue', 'Saved Revenue']
    cost_values = [intervention_cost, lost_revenue, saved_revenue]
    
    fig.add_trace(
        go.Pie(
            labels=cost_labels,
            values=cost_values,
            hole=0.4,
            marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71'])
        ),
        row=1, col=1
    )
    
    #bar chart 
    scenarios = ['Baseline (No Model)', 'With Predictive Model']
    costs = [baseline_cost, model_cost]
    savings = [0, net_savings]
    
    fig.add_trace(
        go.Bar(
            name='Total Cost',
            x=scenarios,
            y=costs,
            marker_color=['#95a5a6', '#3498db'],
            text=[f'Rs.{c:,.0f}' for c in costs],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Net Savings',
            x=['With Predictive Model'],
            y=[net_savings],
            marker_color='#2ecc71',
            text=[f'Rs.{net_savings:,.0f}'],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        template=get_plot_template()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    #detailed breakdown
    st.markdown('<div class="subsection-header">Detailed Breakdown</div>', unsafe_allow_html=True)
    
    breakdown_data = {
        'Metric': [
            'Actual Churners',
            'Predicted Churners (TP + FP)',
            'True Positives',
            'False Positives',
            'False Negatives',
            'Successful Retentions'
        ],
        'Count': [
            actual_churners,
            true_positives + false_positives,
            true_positives,
            false_positives,
            false_negatives,
            int(true_positives * retention_success_rate)
        ],
        'Description': [
            'Customers who would churn without intervention',
            'Customers targeted for retention offers',
            'Correctly identified churners',
            'Loyal customers receiving unnecessary offers',
            'Churners missed by the model',
            'Customers retained through intervention'
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    
    st.dataframe(
        breakdown_df.style.format({'Count': '{:,}'}),
        use_container_width=True,
        height=300
    )
    
    #sensitivity analysis
    st.markdown('<div class="subsection-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        success_rates = np.arange(0.5, 0.95, 0.05)
        savings_by_success = []
        
        for rate in success_rates:
            saved = true_positives * rate * customer_lifetime_value
            net_save = baseline_cost - (intervention_cost + (actual_churners - true_positives) * customer_lifetime_value)
            savings_by_success.append(net_save)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=success_rates*100,
            y=savings_by_success,
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Impact of Retention Success Rate',
            xaxis_title='Retention Success Rate (%)',
            yaxis_title='Net Savings (Rs.)',
            height=300,
            template=get_plot_template()
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        precisions = np.arange(0.5, 0.95, 0.05)
        savings_by_precision = []
        
        for prec in precisions:
            fp_rate = 1 - prec
            fp = int((num_customers - actual_churners) * fp_rate)
            intervention = (true_positives + fp) * retention_offer_cost
            saved = true_positives * retention_success_rate * customer_lifetime_value
            savings_by_precision.append(baseline_cost - (intervention + lost_revenue))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=precisions*100,
            y=savings_by_precision,
            mode='lines+markers',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Impact of Model Precision',
            xaxis_title='Model Precision (%)',
            yaxis_title='Net Savings (Rs.)',
            height=300,
            template=get_plot_template()
        )
        st.plotly_chart(fig, use_container_width=True)
    
    #key takeaways
    st.markdown('<div class="subsection-header">Key Financial Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Return on Investment**
        
        For every Rs.{retention_offer_cost:,} spent on retention offers:
        
        → **Rs.{int(customer_lifetime_value * retention_success_rate / retention_offer_cost):,} saved**
        
        **ROI: {roi:.0f}%**
        """)
    
    with col2:
        st.info(f"""
        **Model Coverage**
        
        Model identifies:
        
        → **{model_recall*100:.1f}%** of actual churners
        
        Misses only **{false_negatives}** high-value customers
        
        **Coverage: Excellent**
        """)
    
    with col3:
        efficiency = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        st.info(f"""
        **Campaign Efficiency**
        
        Retention offers are:
        
        → **{efficiency*100:.1f}%** accurate
        
        **{false_positives}** unnecessary offers
        
        **Efficiency: {'Good' if efficiency > 0.6 else 'Needs Improvement'}**
        """)

# #footer
# st.markdown("---")
# col1, col2, col3 = st.columns([2, 1, 2])
# with col2:
#     st.markdown("""
#     <div style='text-align: center; opacity: 0.6; font-size: 0.85rem;'>
#         E-Commerce Churn Analytics Dashboard
#     </div>
#     """, unsafe_allow_html=True)
