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
    # page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to force light mode and override any dark mode settings
st.markdown("""
<style>
    /* Force light mode by overriding all Streamlit dark mode styles */
    body {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Remove dark mode toggle */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Main styling - Light mode optimized */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
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
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-left: 10px;
        border-left: 4px solid #3498db;
    }
    
    /* Cards with light theme */
    .metric-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .highlight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Fixed Tabs styling - Light theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
        padding: 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        margin-right: 2px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
        color: #34495e;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e3f2fd;
        color: #2c3e50;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
        border-color: #3498db !important;
        font-weight: 600;
    }
    
    /* Plot containers */
    .plot-container {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 10px;
    border-left: 4px solid #3498db;
    border-top: 1px solid #e0e0e0;
    border-right: 1px solid #e0e0e0;
    border-bottom: 1px solid #e0e0e0;
    box-shadow: none;
    margin-bottom: 1.5rem;
}

    
    /* Financial assessment styling */
    .savings-positive {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: #d4edda;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    .savings-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: #f8d7da;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Form elements */
    .stSlider, .stSelectbox, .stNumberInput {
        background-color: white !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.9rem;
        color: #262730 !important;
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
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
        border-bottom: 1px solid #e0e0e0;
    }
    
    .dataframe tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Confusion matrix container */
    .confusion-matrix-container {
        max-width: 600px;
        margin: 0 auto;
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Spacing */
    .spacing {
        margin-top: 2rem;
    }
    
    /* Button styling */
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
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
    }
    
    .stAlert [data-baseweb="notification"] {
        border-radius: 8px;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio > div {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Ensure all text is visible in light mode */
    * {
        color: #262730 !important;
    }
    
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: #262730 !important;
    }
    
    /* Override any Streamlit dark mode text */
    .stMarkdown, .stText, .stWrite {
        color: #262730 !important;
    }
    
    /* Force sidebar text color */
    .css-1d391kg p,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg label {
        color: #262730 !important;
    }
    
    /* Plotly chart background override */
    .js-plotly-plot .plotly {
        background-color: white !important;
    }
    
    /* Table of contents styling */
    .stToc {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Fix for markdown text formatting */
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



[data-testid="stNumberInput"] {
    background-color: transparent !important;
}

[data-testid="stNumberInput"] > div {
    background-color: transparent !important;
    padding: 0 !important;
}

[data-testid="stNumberInput"] > div > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stNumberInput"] input {
    background-color: white !important;
    border: 1px solid #ced4da !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
}


[data-testid="stSlider"] {
    background-color: transparent !important;
}

[data-testid="stSlider"] > div {
    background-color: transparent !important;
    padding: 0 !important;
}

[data-testid="stSlider"] > div > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}


[data-testid="stPlotlyChart"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stPlotlyChart"] > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.modebar {
    background-color: transparent !important;
}


[data-testid="stRadio"] > div {
    background-color: transparent !important;
    padding: 0.5rem 0 !important;
    border: none !important;
}


.stForm {
    background-color: transparent !important;
    border: none !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"],
[data-testid="stSidebar"] [data-testid="stSlider"],
[data-testid="stSidebar"] [data-testid="stRadio"] {
    background-color: transparent !important;
}

/* Remove any default Streamlit widget backgrounds */
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stRadio {
    background-color: transparent !important;
}


[data-testid="stMetricValue"] {
    background-color: transparent !important;
}


/* Force remove all widget containers backgrounds */
[data-baseweb="input"],
[data-baseweb="slider"],
[data-baseweb="radio"] {
    background-color: transparent !important;
}

/* Remove any residual borders */
.st-emotion-cache-* {
    border: none !important;
}



</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_model():
    try:
        with open('churn_model_final.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("Model file not found. Please ensure 'churn_model_final.pkl' is in the same directory.")
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
    
    # Simulate churn based on business rules
    churn_prob = (
        (df['Tenure'] < 5) * 0.4 +
        (df['Complain'] == 1) * 0.3 +
        (df['SatisfactionScore'] < 3) * 0.2 +
        (df['DaySinceLastOrder'] > 7) * 0.1
    )
    df['Churn'] = np.random.binomial(1, churn_prob.clip(0, 1))
    
    return df

# Load model and data
model = load_model()
df = load_sample_data()

# Sidebar navigation with light theme styling
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='margin: 0; font-size: 1.5rem;'>E-Commerce Analytics</h3>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Customer Churn Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select Dashboard Section",
        ["Business Overview", "Model Analysis", "Churn Prediction", "Financial Assessment"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Business parameters
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
    <div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>
        Built for Customer Experience Optimization
    </div>
    """, unsafe_allow_html=True)

# Main content area
if page == "Business Overview":
    st.markdown('<div class="main-header">E Commerce Customer Churn Analytics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Current Churn Rate",
            f"{df['Churn'].mean()*100:.1f}%",
            delta=f"-{5.2:.1f}% target",
            delta_color="inverse"
        )
        # st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        revenue_at_risk = df['Churn'].sum() * customer_lifetime_value
        st.metric(
            "Revenue at Risk",
            f"Rs.{revenue_at_risk:,.0f}",
            delta=f"-Rs.{(revenue_at_risk * 0.3):,.0f}",
            delta_color="inverse"
        )
        # st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Model Coverage",
            "88.9%",
            "Recall Score",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="section-header">Business Context</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    
    with col1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;'>

        <h3>Challenge</h3>
        <p>
        The e commerce platform has experienced significant user growth but faces increasing customer churn.
        Current retention strategies lack personalization and data-driven targeting, reducing their effectiveness.
        </p>

        <h3>Solution</h3>
        <ul>
            <li>Targeted retention campaigns</li>
            <li>Personalized offers and recommendations</li>
            <li>Optimized marketing resource allocation</li>
            <li>Proactive customer engagement</li>
        </ul>

        <h3>Business Impact</h3>
        <ol>
            <li>Improve customer loyalty and lifetime value</li>
            <li>Reduce marketing spend waste</li>
            <li>Increase retention campaign effectiveness</li>
            <li>Enhance overall customer experience</li>
        </ol>

        </div>
        """, unsafe_allow_html=True)
    
    
    with col2:
        # Quick churn distribution
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
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    
    st.markdown('<div class="section-header">Key Metrics Distribution</div>', unsafe_allow_html=True)
    
    # Feature distributions
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
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730')
    )
    
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color='#2c3e50')
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Analysis":
    st.markdown('<div class="main-header">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs with proper spacing
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        " Performance Metrics", 
        " Feature Importance", 
        " Confusion Analysis", 
        " Model Comparison"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Model Performance Summary</div>', unsafe_allow_html=True)
        
        # Simulated model performance metrics
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
                perf_df.style.format({'Value': '{:.3f}'})
                .background_gradient(subset=['Value'], cmap='Blues'),
                use_container_width=True,
                # height=300
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
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#262730')
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="subsection-header">F2-Score Justification</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="markdown-text" style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;'>
        The F2-score (β=2) is optimized as the primary metric because:
        
        1. **Business Priority**: Missing a churning customer (false negative) costs significantly more than 
           an unnecessary retention offer (false positive)
        
        2. **Recall Focus**: Gives twice the weight to recall compared to precision
        
        3. **Cost Sensitivity**: Aligns with business goal of aggressively reducing churn
        
        4. **Resource Optimization**: Enables effective intervention on high-risk customers
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        # Feature importance data
        feature_importance = {
            'Feature': [
                'CashbackAmount', 'WarehouseToHome', 'Tenure', 'DaySinceLastOrder',
                'NumberOfAddress', 'SatisfactionScore', 'NumberOfDeviceRegistered',
                'Complain', 'MaritalStatus_Single', 'MaritalStatus_Married',
                'PreferedOrderCat_Laptop', 'PreferedOrderCat_Fashion',
                'PreferedOrderCat_Mobile', 'PreferedOrderCat_Others',
                'MaritalStatus_Divorced', 'PreferedOrderCat_Grocery'
            ],
            'Importance': [1240, 911, 663, 585, 377, 346, 261, 116, 111, 68, 59, 56, 42, 29, 22, 14]
        }
        
        fi_df = pd.DataFrame(feature_importance)
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(10)
        
        # Create feature importance plot
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
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature insights in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # st.markdown("#### Key Insights")
            st.markdown("""
            ### Key Insights

            - **CashbackAmount**: Most influential feature – incentives strongly affect retention  
            - **WarehouseToHome**: Delivery distance impacts customer satisfaction  
            - **Tenure**: Longer-tenured customers are more loyal  
            - **Recent Activity**: *DaySinceLastOrder* is critical for churn prediction
            """)

        
        with col2:
            st.markdown("#### Business Implications")
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px;">
                <ol>
                    <li><strong>Enhance Loyalty Programs</strong>: Focus on cashback and rewards</li>
                    <li><strong>Optimize Logistics</strong>: Reduce delivery distance impact</li>
                    <li><strong>Early Intervention</strong>: Target new customers with onboarding</li>
                    <li><strong>Proactive Engagement</strong>: Monitor recent purchase patterns</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    
    with tab3:
        st.markdown('<div class="subsection-header">Confusion Matrix Analysis</div>', unsafe_allow_html=True)
        
        # Create a centered confusion matrix
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Simulated confusion matrix
            cm = np.array([[503, 44], [12, 95]])  # Example values
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Not Churn', 'Predicted Churn'],
                y=['Actual Not Churn', 'Actual Churn'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16, "color": "white"},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix - Final Model',
                height=400,
                width=500,
                xaxis_title='Predicted',
                yaxis_title='Actual',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#262730')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown('<div class="subsection-header">Performance Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Positives", tp, "Correct churn predictions", delta_color="off")
        
        with col2:
            st.metric("False Positives", fp, "Incorrect churn predictions", delta_color="off")
        
        with col3:
            st.metric("False Negatives", fn, "Missed churners", delta_color="off")
        
        with col4:
            st.metric("True Negatives", tn, "Correct non-churn predictions", delta_color="off")
        
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
            
            **Acceptable trade-off** for higher recall
            """)
        
        with col2:
            st.error(f"""
            **False Negatives (Cost: Rs.{fn * customer_lifetime_value:,})**
            
            Customers predicted to stay but actually churn.
            
            **Business Impact**:
            - Lost customers
            - Lost future revenue
            - Failed retention opportunity
            
            **Minimized by F2-score optimization**
            """)
    
    with tab4:
        st.markdown('<div class="subsection-header">Model Comparison</div>', unsafe_allow_html=True)
        
        # Model comparison data
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
        
        # Apply styling but don't use background_gradient on string columns
        styled_df = comp_df.style.format({
            'F2-Score': '{:.3f}', 
            'Recall': '{:.3f}', 
            'Precision': '{:.3f}', 
            'Training Time (s)': '{:.1f}'
        })
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            # height=300
        )
        
        # Visualization
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
            textfont=dict(size=10, color='#262730')
        ))
        
        fig.update_layout(
            title='Model Comparison: Precision vs Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tab-content

elif page == "Churn Prediction":
    st.markdown('<div class="main-header">Customer Churn Prediction</div>', unsafe_allow_html=True)
    
    # Prediction interface
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
        
        # Display risk factors with proper HTML formatting
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
        
        # Feature importance reminder
        st.markdown("---")
        st.markdown("#### Top Churn Indicators")
        st.markdown("""
        <div class="markdown-text">
        1. Low cashback amount
        2. Recent complaints
        3. Short customer tenure
        4. High warehouse distance
        5. Recent inactivity
        </div>
        """, unsafe_allow_html=True)
    
    if predict_button:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Prepare input data
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
        
        # Convert to DataFrame for display
        input_df = pd.DataFrame([input_data])
        
        # Display input summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Customer Profile Summary")
            styled_df = input_df.T.rename(columns={0: 'Value'})
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            # Simulate prediction
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
                <div style='text-align: center; padding: 2rem; background-color: #f8d7da; 
                            border-radius: 10px; border: 2px solid #dc3545; margin-top: 20px;'>
                    <h1 style='color: #721c24; margin: 0;'>⚠️</h1>
                    <h2 style='color: #721c24; margin: 1rem 0;'>HIGH CHURN RISK</h2>
                    <p style='color: #721c24; font-size: 1.2rem;'>Probability: {churn_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: #d4edda; 
                            border-radius: 10px; border: 2px solid #28a745; margin-top: 20px;'>
                    <h1 style='color: #155724; margin: 0;'>✅</h1>
                    <h2 style='color: #155724; margin: 1rem 0;'>LOW CHURN RISK</h2>
                    <p style='color: #155724; font-size: 1.2rem;'>Probability: {churn_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Risk gauge
        st.markdown("#### Risk Assessment")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score", 'font': {'size': 24, 'color': '#262730'}},
            number={'suffix': '%', 'font': {'size': 40, 'color': '#262730'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#262730'},
                'bar': {'color': "#e74c3c" if churn_prediction else "#27ae60"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#f8d7da"}
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
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### Retention Recommendations")
        
        if churn_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>
                **Immediate Actions (Within 24 hours):**
                
                - Personal outreach from customer success
                - Special retention offer (20% discount)
                - Priority support assignment
                - Follow-up call scheduled
                
                **Expected Cost:** Rs.{retention_offer_cost:,}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>
                **Strategic Actions (Next 7 days):**
                
                - Review complaint history
                - Analyze purchase patterns
                - Customize future offers
                - Monitor engagement metrics
                
                **Potential Savings:** Rs.{customer_lifetime_value:,}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("""
            **Maintenance Actions:**
            
            - Continue standard engagement
            - Monitor satisfaction scores
            - Regular promotional offers
            - Quarterly check-ins
            
            **No immediate intervention required.** Focus resources on higher-risk customers.
            """)

elif page == "Financial Assessment":
    st.markdown('<div class="main-header">Financial Impact Assessment</div>', unsafe_allow_html=True)
    
    # Simulation controls
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
    
    # Calculate financial impact
    actual_churners = int(num_customers * baseline_churn_rate)
    
    # Using model metrics from analysis
    model_recall = 0.889  # From model analysis
    model_fpr = 0.08  # Estimated false positive rate
    
    # Calculate predictions
    true_positives = int(actual_churners * model_recall)
    false_negatives = actual_churners - true_positives
    false_positives = int((num_customers - actual_churners) * model_fpr)
    
    # Cost calculations
    baseline_cost = actual_churners * customer_lifetime_value
    
    intervention_cost = (true_positives + false_positives) * retention_offer_cost
    lost_revenue = false_negatives * customer_lifetime_value
    saved_revenue = true_positives * retention_success_rate * customer_lifetime_value
    model_cost = intervention_cost + lost_revenue
    
    net_savings = baseline_cost - model_cost
    roi = (net_savings / intervention_cost) * 100 if intervention_cost > 0 else 0
    
    # Display results
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
    
    # Visualization
    st.markdown('<div class="subsection-header">Cost-Benefit Analysis</div>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cost Breakdown', 'Financial Impact'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Pie chart - Cost breakdown
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
    
    # Bar chart - Financial comparison
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
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown - FIXED VERSION
    st.markdown('<div class="subsection-header">Detailed Breakdown</div>', unsafe_allow_html=True)
    
    # Create breakdown data WITHOUT percentage in numeric columns
    breakdown_data = {
        'Metric': [
            'Actual Churners',
            'Predicted Churners (TP + FP)',
            'True Positives',
            'False Positives',
            'False Negatives',
            'Successful Retentions',
            'Retention Success Rate (%)'  # Note: This will be string, not numeric
        ],
        'Count': [
            actual_churners,
            true_positives + false_positives,
            true_positives,
            false_positives,
            false_negatives,
            int(true_positives * retention_success_rate),
            retention_success_rate * 100  # Keep as float for display
        ],
        'Description': [
            'Customers who would churn without intervention',
            'Customers targeted for retention offers',
            'Correctly identified churners',
            'Loyal customers receiving unnecessary offers',
            'Churners missed by the model',
            'Customers retained through intervention',
            'Effectiveness of retention offers'
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    
    # Format the percentage column properly
    breakdown_df_display = breakdown_df.copy()
    breakdown_df_display['Count'] = breakdown_df_display.apply(
        lambda row: f"{row['Count']:.1f}%" if 'Retention Success Rate' in row['Metric'] else f"{int(row['Count']):,}",
        axis=1
    )
    
    # Display the dataframe WITHOUT gradient (which was causing the error)
    st.dataframe(
        breakdown_df_display,
        use_container_width=True,
        height=300
    )
    
    # Sensitivity analysis
    st.markdown('<div class="subsection-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vary retention success rate
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
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vary model precision
        precisions = np.arange(0.5, 0.95, 0.05)
        savings_by_precision = []
        
        for prec in precisions:
            # Simplified calculation
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
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key takeaways
    st.markdown('<div class="subsection-header">Key Financial Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;'>
        
        
        For every Rs.{retention_offer_cost:,} spent on retention offers:
        
        → **Rs.{int(customer_lifetime_value * retention_success_rate / retention_offer_cost):,} saved**
        
        **ROI: {roi:.0f}%**
        </div>
        """, unsafe_allow_html=True)

    
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;'>
        
        
        Model identifies:
        
        → **{model_recall*100:.1f}%** of actual churners
        
        Misses only **{false_negatives}** high-value customers
        
        **Coverage: Excellent**
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        efficiency = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;'>
        
        
        Retention offers are:
        
        → **{efficiency*100:.1f}%** accurate
        
        **{false_positives}** unnecessary offers
        
        **Efficiency: {'Good' if efficiency > 0.6 else 'Needs Improvement'}**
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem; 
                background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;'>
        <p style='margin: 0;'><strong>E-Commerce Customer Churn Prediction Dashboard</strong></p>
        <p style='margin: 0; font-size: 0.8rem;'>Built with Streamlit • Optimized for F₂-Score</p>
    </div>
    """, unsafe_allow_html=True)