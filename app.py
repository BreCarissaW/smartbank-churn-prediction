# Base imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('Set2')

# Modeling imports
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import joblib

# Streamlit imports and Custom Functions
import streamlit as st
from functions import clean_prep_data, load_data, transform_customer_service, transform_online_activity, transform_transaction_history, merge_data, clean_prep_data

# Load the trained model pipeline
model = joblib.load('outputs/xgb_model.pkl')

# Title
st.title('SmartBank Churn Prediction App Demo - For Internal Use Only')
st.write('Upload the required CSV files to predict customer churn probabilities and identify high-risk customers.')

# File uploader for the 4 required CSV files
cust_dem_upload = st.file_uploader('Upload Customer Demographics CSV:', type='csv')
cust_service_upload = st.file_uploader('Upload Customer Service CSV:', type='csv')
online_activity_upload = st.file_uploader('Upload Online Activity CSV:', type='csv')
transaction_history_upload = st.file_uploader('Upload Transaction History CSV:', type='csv')

# Create a function to handle the prediction logic
def predict_churn(files):
    # Check if all 4 files are uploaded
    if len(files) != 4:
        st.error("Please upload all 4 required CSV files.")
        return
    
    # Load and merge data
    cust_dem_data = pd.read_csv(cust_dem_upload)
    cust_service_data = transform_customer_service(pd.read_csv(cust_service_upload))
    online_activity_data = transform_online_activity(pd.read_csv(online_activity_upload))
    transaction_history_data = transform_transaction_history(pd.read_csv(transaction_history_upload))
    data = merge_data(cust_dem_data, cust_service_data, online_activity_data, transaction_history_data)

    # Clean and engineer features
    CustomerID = data['CustomerID'].copy()      # Save CustomerID for deployment
    cleaned_data = clean_prep_data(data)
    cleaned_data = cleaned_data.drop(columns=['CustomerID'], errors='ignore')

    # Make predictions using loaded model
    probabilities = model.predict_proba(cleaned_data)[:, 1]
    
    # Create results dataframe
    all_results = pd.DataFrame({'CustomerID': CustomerID, 'ChurnProbability': probabilities.round(2)})
    churners = (
        all_results[all_results['ChurnProbability'] >= 0.5]
        .sort_values(by='ChurnProbability', ascending=False)
        .reset_index(drop=True)
    )
    return all_results, churners


# Run prediction when button is clicked
if st.button("Predict Churn"):
    if cust_dem_upload and cust_service_upload and online_activity_upload and transaction_history_upload:
        prediction_output = predict_churn([cust_dem_upload, cust_service_upload, online_activity_upload, transaction_history_upload])

        if prediction_output is None:
            st.stop()

        results_df, churners_df = prediction_output

        if results_df is not None:
            st.header('Predictions')
            st.subheader('High-Risk Customers (Churn Probability >= 0.5)')
            st.write(f"Total High-Risk Customers: {len(churners_df)}")
            st.write('(Model Metrics: Model: XGBoost | ROC-AUC: 0.52 | Precision: 0.42 | Recall: 0.20 | F1-Score: 0.27)')

            # Visualize the distribution of churn probabilities
            plot_data = results_df.copy()
            plot_data['ChurnStatus'] = plot_data['ChurnProbability'].apply(lambda x: 'High Risk' if x >= 0.5 else ('Moderate Risk' if x >= 0.3 else 'Low Risk'))

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,18))

            # Countplot with hue to show the distribution of churn status
            sns.countplot(data=plot_data, x='ChurnStatus', hue='ChurnStatus', ax=axs[0])
            for container in axs[0].containers:     # Loop needed when hue is used.
                axs[0].bar_label(container, fmt='%d', label_type='edge', fontsize=20)
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=16)
            axs[0].set_title('Distribution of Churn Status', fontsize=20)

            # Pie chart to show the proportion of churned vs non-churned customers
            counts = plot_data['ChurnStatus'].value_counts()
            axs[1].pie(counts, labels=counts.index, autopct='%0.2f%%', textprops={'fontsize': 20})

            fig.savefig('churn_plots.png')  # Save the plot as a PNG file for download

            # Display the table and plots side by side
            col1, col2 = st.columns([1.4, 1])

            with col1:
                
                st.dataframe(churners_df, use_container_width=True)

            with col2:
                st.pyplot(fig, use_container_width=True)

            # Provide option to download results as CSV
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            churners_csv = churners_df.to_csv(index=False).encode('utf-8')

            col3, col4, col5 = st.columns(3)

            # Download button to download all results as a CSV file
            with col3:
                st.download_button(
                    label="Download All Results as CSV",
                    data=csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            # Download button to download the churners as a CSV file
            with col4:
                st.download_button(
                    label="Download Churners as CSV",
                    data=churners_csv,
                    file_name="churners.csv",
                    mime="text/csv"
                )

            # Button to download plot as PNG
            with col5:
                with open('churn_plots.png', 'rb') as f:
                    st.download_button(
                        label="Download Churn Plots as PNG",
                        data=f,
                        file_name="churn_plots.png",
                        mime="image/png"
                    )
        else:
            st.error("Please upload the 4 required CSV files first.")

