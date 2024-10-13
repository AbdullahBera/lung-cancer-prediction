import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Set wide mode for Streamlit app layout
st.set_page_config(layout="wide")

# Custom CSS to adjust layout, reduce predictor checkbox size, and add space between sections
st.markdown("""
    <style>
        .custom-title {
            font-size: 14px;  /* Reduced font size */
            font-weight: bold;
            margin-top: 5px;  /* Minimized margin */
        }
        .stSlider {
            margin-top: -10px;  /* Adjusted space above the slider */
        }
        .custom-probability {
            font-size: 20px;  /* Larger font for the probability at the top */
            font-weight: bold;
            color: #d9534f;  /* Red color for visibility */
        }
        .custom-space {
            margin-top: 20px;  /* Added space between Probability and Select Age */
        }
        .stCheckbox > label {
            font-size: 12px;  /* Smaller font size for the checkboxes */
            padding-left: 5px;  /* Reduce space before the label */
        }
        .stCheckbox {
            margin-bottom: -5px;  /* Reduce space between checkboxes */
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/kalu/Desktop/Lin-Reg-project/survey_lung_cancer.csv')
    df.columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                  'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
                  'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                  'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                  'CHEST PAIN', 'LUNG_CANCER']
    df = df.replace({1: 0, 2: 1})

    # Encode categorical variables
    df["GENDER"] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, "NO": 0})
    return df

# Train the model
def train_model(df):
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(solver='liblinear', random_state=24)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# Load data and train model
df = load_data()
model, X_train, X_test, y_train, y_test = train_model(df)

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = 'Predictor-Dependent Analysis'

# Sidebar for navigation using buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Predictor-Dependent Analysis", key="predictor"):
    st.session_state.page = 'Predictor-Dependent Analysis'
if st.sidebar.button("Non-Predictor Dependent Analysis", key="non_predictor"):
    st.session_state.page = 'Non-Predictor Dependent Analysis'

# Predictor-Dependent Analysis Page
if st.session_state.page == 'Predictor-Dependent Analysis':
    
    # Probability of Lung Cancer at the top
    st.markdown(f'<div class="custom-probability">Probability of Lung Cancer: 79.47%</div>', unsafe_allow_html=True)

    # Space between Probability of Lung Cancer and Select Age
    st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)

    # Age slider (without the "Age Slider" title)
    age_value = st.slider("Select Age", int(df['AGE'].min()), int(df['AGE'].max()), int(df['AGE'].mean()))

    # Set up horizontal checkboxes for feature selection
    st.markdown('<div class="custom-title">Select Features to Analyze their Impact</div>', unsafe_allow_html=True)

    predictors = [col for col in df.columns[:-1] if col != 'AGE']

    # Create a list of selected predictors using checkboxes arranged horizontally
    selected_predictors = []
    num_cols = 4  # Number of checkboxes per row
    rows = [predictors[i:i + num_cols] for i in range(0, len(predictors), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for col, predictor in zip(cols, row):
            if col.checkbox(predictor, key=predictor):
                selected_predictors.append(predictor)

    # Display selected predictors
    if selected_predictors:
        st.write(f"Currently selected predictors: {', '.join(selected_predictors)}")
    else:
        st.write("No predictors selected. Please select predictors to analyze their impact.")
    
    # Calculate probability of lung cancer based on selected features
    input_data = {}
    for feature in df.columns[:-1]:  # Loop through all features (except target)
        if feature in selected_predictors:
            input_data[feature] = 1  # Set selected features to 1 (checked)
        else:
            input_data[feature] = 0  # Default to 0 (not selected)

    # Adding AGE slider value to input data
    input_data['AGE'] = age_value

    # Convert input data into DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Ensure that the input_df columns are in the same order as during training
    input_df = input_df[df.columns[:-1]]  # Reorder to match the training feature order

    # Make prediction and display as a percentage
    prediction_percentage = model.predict_proba(input_df)[0][1] * 100  # Convert to percentage

    # Interactive Feature Impact Plot (impact of coefficients without "Interactive Feature Impact" title)
    fig = go.Figure()

    # Display the impact of coefficients for selected features
    for feature in selected_predictors:
        feature_values = np.linspace(df[feature].min(), df[feature].max(), 100)
        
        # Impact is the coefficient scaled by feature values
        impact = model.coef_[0][df.columns[:-1].get_loc(feature)] * feature_values

        fig.add_trace(go.Scatter(x=feature_values, y=impact, mode='lines', name=feature))

    fig.update_layout(
        title="Impact of Coefficients",  # Updated the title to reflect coefficient impact
        xaxis=dict(title="Feature Value", titlefont=dict(size=14)),  # Reduced font size for labels
        yaxis=dict(title="Impact Value (based on coefficients)", titlefont=dict(size=14)),  # Reduced font size for labels
        showlegend=True,
        height=350  # Reduced plot height to minimize white space
    )

    st.plotly_chart(fig)

# Non-Predictor Dependent Analysis Page
elif st.session_state.page == 'Non-Predictor Dependent Analysis':
    st.markdown('<div class="custom-title">Non-Predictor Dependent Model Analysis</div>', unsafe_allow_html=True)

    # Hide the sidebar input features when on this page
    st.sidebar.empty()

    # Use st.columns to show the graphs side by side
    col1, col2 = st.columns(2)

    with col1:
        # Feature Importance with smaller height
        st.markdown('<div class="custom-title">Feature Importance</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_[0]})
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
        fig_imp = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', height=350)  # Reduced height
        fig_imp.update_layout(
            xaxis=dict(title="Coefficient", titlefont=dict(size=14)),  # Reduced font size
            yaxis=dict(title="Feature", titlefont=dict(size=14))  # Reduced font size
        )
        st.plotly_chart(fig_imp)

    with col2:
        # ROC Curve with smaller height
        st.markdown('<div class="custom-title">ROC Curve</div>', unsafe_allow_html=True)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_curve_fig = go.Figure()
        roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
        roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
        roc_curve_fig.update_layout(
            xaxis=dict(title="False Positive Rate", titlefont=dict(size=14)),  # Reduced font size
            yaxis=dict(title="True Positive Rate", titlefont=dict(size=14)),  # Reduced font size
            height=350  # Reduced plot height
        )
        st.plotly_chart(roc_curve_fig)

    # F1 Score vs Number of Predictors
    st.markdown('<div class="custom-title">F1 Score vs Number of Predictors</div>', unsafe_allow_html=True)

    # Simulating a DataFrame for demonstration
    result_df = pd.DataFrame({
        'Number of Predictors': np.arange(1, 21),
        'F1 Score': np.random.rand(20)
    })

    # Calculate the maximum F1 Score for each number of predictors
    f1_by_predictor_count = result_df.groupby('Number of Predictors')['F1 Score'].max().reset_index()

    # Filter the data for number of predictors between 8 and 12
    filtered_data = f1_by_predictor_count[(f1_by_predictor_count['Number of Predictors'] >= 8) &
                                          (f1_by_predictor_count['Number of Predictors'] <= 12)]

    # Plotting the filtered data
    plt.figure(figsize=(4, 2.5))  # Reduced size for F1 Score plot
    plt.plot(filtered_data['Number of Predictors'], filtered_data['F1 Score'], marker='o')
    plt.xlabel('Number of Predictors', fontsize=10)  # Reduced font size for xlabel
    plt.ylabel('F1 Score', fontsize=10)  # Reduced font size for ylabel
    plt.title('F1 Score vs. Number of Predictors (8 to 12)', fontsize=12)  # Reduced title size
    plt.grid(True)

    # Display the plot in the Streamlit app
    st.pyplot(plt)

    # Clear the figure to avoid overlapping plots if running multiple times
    plt.clf()