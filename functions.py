import numpy as np
import pandas as pd
from xgboost import data

def load_data(customer_demo_file, customer_service_file, online_activity_file, transaction_history_file):
    '''
    Load the four CSV files into pandas DataFrames.
    
    Parameters:
        demo_file: File-like object for Customer Demographics CSV
        service_file: File-like object for Customer Service CSV
        activity_file: File-like object for Online Activity CSV
        transaction_file: File-like object for Transaction History CSV
    '''
    customer_demographics = pd.read_csv(customer_demo_file)
    customer_service = pd.read_csv(customer_service_file)
    online_activity = pd.read_csv(online_activity_file)
    transaction_history = pd.read_csv(transaction_history_file)

    return customer_demographics, customer_service, online_activity, transaction_history


def transform_customer_service(customer_service):
    """
    Transforms the highly granular customer_service dataframe by:
     1. Encoding ResolutionStatus as binary,
     2. Grouping by CustomerID,
     3. Replacing individual interaction dates with DaysSinceLastInteraction and,
     4. Calculating new column ResolutionRate.
    """
    customer_service = pd.get_dummies(customer_service, columns=['InteractionType'], drop_first=True, dtype=int)

    # Binarize resolution status for grouping 
    customer_service['ResolutionStatus'] = customer_service['ResolutionStatus'].replace({'Unresolved': 0, 'Resolved': 1})

    # Convert InteractionDate to datetime
    customer_service['InteractionDate'] = pd.to_datetime(customer_service['InteractionDate'])

    # Group customer_service by CustomerID
    customer_service = customer_service.groupby('CustomerID').agg({
                                                'InteractionID': 'nunique',           
                                                'InteractionDate': 'max',
                                                'ResolutionStatus': 'sum',
                                                'InteractionType_Feedback': 'sum',
                                                'InteractionType_Inquiry': 'sum'
                                                }).reset_index().rename(columns={
                                                'InteractionID': 'NumInteractions', 
                                                'ResolutionStatus': 'InteractionsResolved'}) 

    # Calculate DaysSinceLastInteraction
    max_int_date = customer_service['InteractionDate'].max()
    customer_service['DaysSinceLastInteraction'] = (max_int_date - customer_service['InteractionDate']).dt.days

    # Creating a new column to indicate the rate of interactions resolved
    customer_service['ResolutionRate'] = round(customer_service['InteractionsResolved'] / customer_service['NumInteractions'], 2)

    customer_service['PctInteractionType_Feedback'] = (customer_service['InteractionType_Feedback'] / customer_service['NumInteractions']).round(2)

    customer_service['PctInteractionType_Inquiry'] = (customer_service['InteractionType_Inquiry'] / customer_service['NumInteractions']).round(2)

    # Drop raw date column
    customer_service.drop(columns=['InteractionDate', 'InteractionsResolved', 'InteractionType_Feedback', 'InteractionType_Inquiry'], inplace=True)     

    return customer_service


def transform_online_activity(online_activity):
    """    
    Transforms online_activity dataframe by:
     1. Grouping by CustomerID,
     2. Replacing LastLoginDate with DaysSinceLastLogin and,
     3. Encoding ServiceUsage as dummy variables.
     """
    
    # Get dummies for ServiceUsage so we can aggregate later if needed
    online_activity = pd.get_dummies(online_activity, columns=['ServiceUsage'], drop_first=True, dtype=int)

    # Convert LastLoginDate to datetime
    online_activity['LastLoginDate'] = pd.to_datetime(online_activity['LastLoginDate'])

    # Group online_activity by CustomerID
    online_activity = online_activity.groupby('CustomerID').agg({'LastLoginDate': 'max',
                                                                 'LoginFrequency': 'max',
                                                                 'ServiceUsage_Online Banking': 'sum',
                                                                 'ServiceUsage_Website': 'sum'}).reset_index()

    # Add column for days since last login instead of using raw dates
    max_log_date = online_activity['LastLoginDate'].max()
    online_activity['DaysSinceLastLogin'] = (max_log_date - online_activity['LastLoginDate']).dt.days

    online_activity['UsageRate_OnlineBanking'] = (online_activity['ServiceUsage_Online Banking'] / online_activity['LoginFrequency']).round(2)
    online_activity['UsageRate_Website'] = (online_activity['ServiceUsage_Website'] / online_activity['LoginFrequency']).round(2)

    online_activity.drop(columns=['LastLoginDate', 'ServiceUsage_Online Banking', 'ServiceUsage_Website'], inplace=True)   

    return online_activity


def transform_transaction_history(transaction_history):
    """ 
    Transforms the highly granular transaction_history dataframe by:
     1. Getting dummies for ProductCategory,
     2. Grouping by CustomerID and,
     3. Replacing TransactionDate with DaysSinceLastTransaction.
    """
    # Get dummies for ProductCategory so we can aggregate later if needed
    transaction_history = pd.get_dummies(transaction_history, columns=['ProductCategory'], drop_first=True, dtype=int)

    # Convert TransactionDate to datetime
    transaction_history['TransactionDate'] = pd.to_datetime(transaction_history['TransactionDate'])
    
    # Group transaction_history by CustomerID
    transaction_history = transaction_history.groupby('CustomerID').agg({
                                                    'TransactionID': 'nunique',
                                                    'AmountSpent': 'sum',
                                                    'TransactionDate': 'max',
                                                    'ProductCategory_Clothing': 'sum',
                                                    'ProductCategory_Electronics': 'sum',
                                                    'ProductCategory_Furniture': 'sum',
                                                    'ProductCategory_Groceries': 'sum'
                                                    }).reset_index().rename(columns={'TransactionID': 'NumTransactions'})

    # Add column for days since last transaction instead of using raw dates
    max_date = transaction_history['TransactionDate'].max()
    transaction_history['DaysSinceLastTransaction'] = (max_date - transaction_history['TransactionDate']).dt.days

    for prodcat in ['ProductCategory_Clothing', 'ProductCategory_Electronics', 'ProductCategory_Furniture', 'ProductCategory_Groceries']:
        transaction_history[f'Pct{prodcat}'] = (transaction_history[prodcat] / transaction_history['NumTransactions']).round(2)

    transaction_history.drop(columns=['TransactionDate', 'ProductCategory_Clothing', 'ProductCategory_Electronics', 'ProductCategory_Furniture', 'ProductCategory_Groceries'], inplace=True)     # Drop raw date column
    
    return transaction_history


def merge_data(customer_demographics, customer_service, online_activity, transaction_history):
    '''
    Merge the four dataframes on CustomerID.
    
    Parameters:
        customer_demographics: DataFrame of customer demographics
        customer_service: DataFrame of customer service interactions
        online_activity: DataFrame of online activity
        transaction_history: DataFrame of transaction history
    
    Returns:
        Merged DataFrame with one row per CustomerID
    '''
    # Merge demographics with service data
    data = customer_demographics.merge(customer_service, on='CustomerID', how='left')
    # Merge with online activity
    data = data.merge(online_activity, on='CustomerID', how='left')
    # Merge with transaction history
    data = data.merge(transaction_history, on='CustomerID', how='left')
    return data


def clean_prep_data(data):
    '''
    Clean and engineer features for the merged data.
    
    Parameters:
        data: Merged DataFrame with all customer data

    Returns:
        Cleaned and feature-engineered DataFrame
    '''
# Converting categorical columns to appropriate types
    for col in ['Gender', 'MaritalStatus']:
        data[col] = data[col].astype('category')
    
    # Converting IncomeLevel to ordered categorical
    data['IncomeLevel'] = pd.Categorical(data['IncomeLevel'], categories=['Low', 'Medium', 'High'], ordered=True)

    return data

