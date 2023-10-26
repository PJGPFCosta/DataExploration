import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def most_importante_features_correlation(df,target,num_features,list_to_drop):
    df=df.drop(columns=list_to_drop)
    #correlation
    correlation = df.corr()
    target_correlation = correlation[target].abs().sort_values(ascending=False)
    top_features_correlation = target_correlation[1:num_features+1].index.tolist()  # Drop target and get the N variables with higher corr
    return top_features_correlation

def most_importante_features_treebased(df,target,num_features,list_to_drop):
    #drop variavels like Id,...
    df=df.drop(columns=list_to_drop)

    # Separate categorical and numerical columns
    numerical_cols= [col for col in df.columns if df[col].dtype == 'int64']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]


    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Assuming 'target_variable' is the column you're trying to predict
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]



    # Initialize a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Create a DataFrame to store feature names and their importance scores
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Select the top N features
    top_features_treebased = feature_importance_df.head(num_features)['Feature']

    return top_features_treebased

def most_important_features(df,target,num_features,list_to_drop):
    
    """
    Returns the column names of the N most important variables in a DataFrame.
    By ensembling the results of correlation analysis and TreeBased feature importance.

    Args:
        df: Input DataFrame.
        targe: Input Targe column.
        num_features: Input Number most importante features 
        list_to_drop: List of columns names that are not necessary for ex: Ids,...

    Returns:
        list: List of the most important features.
    """
    

    #correlation
    top_features_correlation=most_importante_features_correlation(df,target,num_features,list_to_drop)
    # tree based
    top_features_treebased=most_importante_features_treebased(df,target,num_features,list_to_drop)

   
    # Convert lists to sets and find the intersection
    common_values = set(top_features_correlation).intersection(top_features_treebased)

    # Convert the result back to a list
    common_values_list = list(common_values)
    # Insert the new value at the beginning
    common_values_list.insert(0, target)

    return common_values_list


def return_categorical_numerical_columns(df,cat_col_forced=[],max_value_for_categorical=20):


    """
    Returns the column names of the numerical and categorical columns.
    You can pass the names of the categorical columns that are like "1,2,3,4,.." and it represents levels 

    Args:
        df: Input DataFrame.
        cat_col_forced: Input list of the categorical columns.
        max_value_for_categorical: Input Number of values in the column that can be considerer categorical and not numeric
        

    Returns:
        list: List of categorical and numerical columns. [categorical_cols,numerical_cols]
    """
    # Separate categorical and numerical columns

    numerical_cols= [col for col in df.columns if df[col].dtype in ["int64", "float64"] and col not in cat_col_forced and df[col].nunique() > max_value_for_categorical]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]

    return categorical_cols,numerical_cols