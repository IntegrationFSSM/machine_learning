import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, target_column='survived', test_size=0.2, random_state=42):
    """
    Prepares the Titanic dataset for training.
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical variables
    - Splits into Train/Test
    """
    
    print("Preprocessing data...")
    
    # Drop columns with too many missing values or irrelevant for initial model
    # cabin has many missing. data.boat, body, home.dest are leaky or irrelevant
    drop_cols = ['cabin', 'boat', 'body', 'home.dest', 'name', 'ticket']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    
    # Identify numerical and categorical columns
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']
    
    # Ensure columns exist (OpenML names might differ slightly, usually lowercase)
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    # Define Preprocessing Steps
    
    # Numerical: Impute with median, Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: Impute with mode (most_frequent), OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Fit and transform
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after encoding (for feature importance)
    # This can be tricky with Pipelines, but useful for interpretation
    try:
        num_names = numeric_features
        cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        feature_names = np.r_[num_names, cat_names]
    except:
        feature_names = None

    return X_train_processed, X_test_processed, y_train, y_test, feature_names
