# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew, kurtosis

# For exploring missing values
import missingno as msno

def missing_values_processing(df):
    # Drop columns with more than 50% missing values
    df.drop(columns='MiscFeature', inplace=True)
    df.drop(columns='Street', inplace=True)

    # Replace NaN values in qualitatives with "No"
    df['PoolQC'] = df['PoolQC'].fillna("No")
    df['Alley'] = df['Alley'].fillna("No")
    df['Fence'] = df['Fence'].fillna("No")
    df['MasVnrType'] = df['MasVnrType'].fillna("No")
    df['BsmtQual'] = df['BsmtQual'].fillna("No")
    df['BsmtCond'] = df['BsmtCond'].fillna("No")
    df['BsmtExposure'] = df['BsmtExposure'].fillna("No")
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna("No")
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna("No")
    df['GarageType'] = df['GarageType'].fillna("No")
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna("No")  
    df['GarageFinish'] = df['GarageFinish'].fillna("No")
    df['GarageQual'] = df['GarageQual'].fillna("No")
    df['GarageCond'] = df['GarageCond'].fillna("No")
 
    # Replace missing values in 'Electrical' with the mode
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])    
    
    # Replace NaN values in quantitatives with 0
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    # Impute missing values in 'LotFrontage' with cross-wise median imputation
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    return df

def features_engineering(df):
    ### Create a new feature 'FireplaceFeature' based on 'Fireplaces' and 'FireplaceQu'

    # Map 'FireplaceQu' to numerical scores
    quality_mapping = {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "No": 0  # Missing values or no fireplace
    }

    df['FireplaceQu'] = df['FireplaceQu'].fillna("No")
    df['FireplaceQu_Score'] = df['FireplaceQu'].map(quality_mapping)
    df['FireplaceFeature'] = df['Fireplaces'] * df['FireplaceQu_Score']
    df.drop(columns=['FireplaceQu', 'Fireplaces', 'FireplaceQu_Score'], inplace=True)

    return df


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/train.csv')

    # Data processing
    print("Shape of the intial dataset: ", df.shape)
 
    print("Missing values processing...")
    df = missing_values_processing(df)
   
    print("Features engineering...")
    df = features_engineering(df)

    print("Shape of the processed dataset: ", df.shape)
    # Save the processed dataset
    df.to_csv('data/processed_data/train_processed.csv', index=False)
