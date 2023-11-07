import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder



class FeatureSelector:

    def __init__(self):
        """
        Methods for Feature Selection
        """
        self=self



    @staticmethod
    def return_categorical_numerical_columns(df,list_to_not_include=[],cat_col_forced=[],max_value_for_categorical=20):


        """
        Returns the column names of the numerical and categorical columns.
        You can pass the names of the categorical columns that are like "1,2,3,4,.." and it represents levels 

        Args:
            df: Input DataFrame.
            list_to_not_include: Input if need, its a list of columns to not include in the split 
            cat_col_forced: Input list of the categorical columns.
            max_value_for_categorical: Input Number of values in the column that can be considerer categorical and not numeric
            

        Returns:
            list: List of categorical and numerical columns and text_cols. [categorical_cols,numerical_cols,text_cols]
        """
        # Separate categorical and numerical columns
        # Check if the values of the column are "free text" and the number of unique values is bigger than the max for categorical
        text_cols= [col for col in df.columns if df[col].dtype == 'object' and col not in cat_col_forced and df[col].nunique() > max_value_for_categorical and col not in list_to_not_include]
        numerical_cols= [col for col in df.columns if df[col].dtype in ["int64", "float64"] and col not in cat_col_forced and df[col].nunique() > max_value_for_categorical and col not in list_to_not_include]
        categorical_cols = [col for col in df.columns if col not in numerical_cols and col not in text_cols and col not in list_to_not_include]

        return categorical_cols,numerical_cols,text_cols
    
    @staticmethod
    def most_importante_features_correlation(df,target,num_features,list_to_drop):
        """
        Returns the column names of the N most important variables in a DataFrame.
        By ensembling the results of correlation analysis and TreeBased feature importance.

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            num_features: Input Number most importante features 
            list_to_drop: List of columns names that are not necessary for ex: Ids,...

        Returns:
            list: List of the most important features and the df of this features with the corresponding importance.
        """

        df=df.drop(columns=list_to_drop)
        #correlation
        correlation = df.corr()
        target_correlation = correlation[target].abs().sort_values(ascending=False)
        top_features_correlation = target_correlation[1:num_features+1].index.tolist()  # Drop target and get the N variables with higher corr
        top_features_correlation_values = target_correlation[1:num_features+1].reset_index() # get all values order
        return top_features_correlation,top_features_correlation_values

    @staticmethod
    def most_importante_features_treebased(df,target,num_features,list_to_drop):
        """
        Returns the column names of the N most important variables in a DataFrame.
        By ensembling the results of correlation analysis and TreeBased feature importance.

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            num_features: Input Number most importante features 
            list_to_drop: List of columns names that are not necessary for ex: Ids,...

        Returns:
            list: List of the most important features and the df of this features with the corresponding importance.
        """
        categorical_cols,numerical_cols,text_columns=FeatureSelector.return_categorical_numerical_columns(df,[target])

        #drop variavels like Id,... and the text columns like Names, ticket names
        list_to_drop.extend(text_columns)
        df=df.drop(columns=list_to_drop)

        # Assuming categorical_cols is a list of categorical column names
        label_encoder = LabelEncoder()
        df_encoded=df
        for col in categorical_cols:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
       
        # Fill Nan values
        df_encoded = df_encoded.fillna(df_encoded.median())

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
        top_features_treebased = feature_importance_df.head(num_features)['Feature'].to_list()

        top_feautres_df=feature_importance_df.head(num_features)

        return top_features_treebased,top_feautres_df
    
    @staticmethod
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
            list: List of the most important features and the df of this features with the corresponding importance.
        """
        

        #correlation
        top_features_correlation,df_features_correlation=FeatureSelector.most_importante_features_correlation(df,target,num_features,list_to_drop)
        # tree based
        top_features_treebased,df_features_treebased=FeatureSelector.most_importante_features_treebased(df,target,num_features,list_to_drop)


        # Get column names from df1
        new_column_names = df_features_treebased.columns.tolist()

        # Rename columns of df2
        df_features_correlation.columns = new_column_names

        concatenated_df = pd.concat([df_features_treebased, df_features_correlation], axis=0)
        concatenated_df = concatenated_df.drop_duplicates(subset=[concatenated_df.columns[0]])

        common_values_list=concatenated_df.iloc[:, 0].to_list()


        return common_values_list,concatenated_df

class Visualization:
    def __init__(self):
            """
            Methods for ploting and visualize the data
            """
            self=self

    @staticmethod
    def plot_correlation(df,target,list_of_features):

        """
        Plots Correlation Matrix all the variables passed with the target.
        Should Be used with only the most important features.

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            list_of_features: List of features that want to be ploted agains the target column. 

        Returns:
            Nothing, it plots
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Concatenate the target column with the top features
        selected_columns = [target] + list_of_features
        selected_df = df[selected_columns]
        
        # Calculate the correlation matrix
        correlation_matrix = selected_df.corr()
    
        size=len(selected_columns)
        
        # Create a heatmap
        plt.figure(figsize=(int(size*2), int(size*1.5)))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Top Features')
        plt.show()

    @staticmethod
    def plot_relationship_cagorical_target_continuous(df,target,cat_cols):
        """
        Plots BoxPlot all the variables passed with the target type continuous.
        Should Be used with only the most important features and only categorical features

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            cat_cols: List of features that want to be ploted agains the target column. 

        Returns:
            Nothing, it plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(cat_cols), figsize=(6*len(cat_cols), 5))

        # Iterate through the categorical columns
        for i, var in enumerate(cat_cols):
            data = pd.concat([df[target], df[var]], axis=1)
            
            # Create a box plot
            sns.boxplot(x=var, y=target, data=data, ax=axes[i])
            #axes[i].set_ylim(0, 800000)  # Set y-axis limits

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()


    @staticmethod
    def plot_relationship_numerical_target_continuous(df,target,num_cols):

        """
        Plots ScatterPlot all the variables passed with the target type continuous.
        Should Be used with only the most important features and only continuous features

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            num_cols: List of features that want to be ploted agains the target column. 

        Returns:
            Nothing, it plots
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(6*len(num_cols), 5))

        # Iterate through the categorical columns
        for i, var in enumerate(num_cols):
            data = pd.concat([df[target], df[var]], axis=1)
            
            # Create a box plot
            sns.scatterplot(x=var, y=target, data=data, ax=axes[i])
            #axes[i].set_ylim(0, 800000)  # Set y-axis limits

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()


    @staticmethod
    def plot_relationship_categorical_target_binary(df, target, cat_cols):

        """
        Plots Barplot all the variables passed with the target type binary.
        Should Be used with only the most important features and only categorical features

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            cat_cols: List of features that want to be ploted agains the target column. 

        Returns:
            Nothing, it plots
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(8, 6*len(cat_cols)))

        # Iterate through the categorical columns
        for i, var in enumerate(cat_cols):
            data = pd.concat([df[target], df[var]], axis=1)

            # Calculate the count of each category for each target class
            count_data = data.groupby([var, target]).size().reset_index(name='count')

            # Create a grouped bar plot
            sns.barplot(x=var, y='count', hue=target, data=count_data, ax=axes[i])
            axes[i].set_title(f'Relationship between {var} and {target}')

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

    @staticmethod
    def plot_relationship_numerical_target_binary(df, target, num_cols):
        
        """
        Plots BoxPlot all the variables passed with the target type binary.
        Should Be used with only the most important features and only continuous features

        Args:
            df: Input DataFrame.
            targe: Input Targe column.
            cat_cols: List of features that want to be ploted agains the target column. 

        Returns:
            Nothing, it plots
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(6*len(num_cols), 5))

        # Iterate through the numerical columns
        for i, var in enumerate(num_cols):
            data = pd.concat([df[target], df[var]], axis=1)

            # Create a box plot or violin plot
            sns.boxplot(x=target, y=var, data=data, ax=axes[i])

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

class Transformations:
    def __init__(self):
            """
            Methods for transforming data to deal with Skewness and Kurtosis
            """
            self=self

    @staticmethod
    def understand_tranformation_lambda(variable_name,transformed_data, lambda_value):
        
        import numpy as np
        print(f"Transformed Data (with lambda={lambda_value}): {transformed_data}")

        if lambda_value<0:
            lambda_value=int(lambda_value)

        if lambda_value == 0:
            explanation = f"The '{variable_name}' variable was log-transformed."
        elif lambda_value == 1:
            explanation = f"No significant transformation was applied to the '{variable_name}' variable."
        elif lambda_value > 1:
            explanation = f"The '{variable_name}' variable was positively transformed with a lambda value of {lambda_value:.2f}."
        elif 0 < lambda_value < 1:
            explanation = f"The '{variable_name}' variable was negatively transformed with a lambda value of {lambda_value:.2f}."
        else:
            explanation = f"The transformation of the '{variable_name}' variable cannot be easily interpreted."

        print(explanation)

    @staticmethod
    def transform_skewness(df,var,num=True,flag_prints=False):        
        """
        Decide and deal with skewness
                
        Positive Skewness:
        - Log Transformation:
            - Works well for data with positive skewness.
            - Reduces the effect of extreme values.
            - Helps in stabilizing variance.
        - Square Root Transformation:
            - Similar to log transformation, it reduces the impact of extreme values.
            - Useful for right-skewed data.

        Negative Skewness:
        - Cube Root Transformation:
            - Works well for data with negative skewness.
            - It's a milder transformation compared to the square root.
        - Square Transformation:
            - Similar to cube root transformation, it can be used to reduce negative skewness.


        Kurtosis:
        Kurtosis measures the tailedness of a probability distribution. It tells you about the shape of the distribution's tails in relation to its peak.

        - Positive Kurtosis (Leptokurtic): The distribution has heavy tails and a high peak. It indicates more outliers than a normal distribution.
        - Negative Kurtosis (Platykurtic): The distribution has lighter tails and a lower peak. It indicates fewer outliers than a normal distribution.

        Args:
            df: Input DataFrame.
            var: Input Targe column.
            flag_prints: Input boolean to print or not

        Returns:
            which transformation is best
        """

        import numpy as np
        from scipy.stats import boxcox,yeojohnson,ks_2samp
        
        # Define a metric based on skewness and kurtosis differences
        def metric(skew, kurt,flag_prints=False,skew_standart=0.5,kurt_stabdart=3):

            skewnewss=skew_standart-abs(skew)
            kurtosis=kurt_stabdart-abs(kurt)
            # Kolmogorov-Smirnov statistic
            if flag_prints:
                print("skew: ",skew, " - skewnewss: ",skewnewss)
                print("kurt: ",kurt, " - kurtosis: ",kurtosis)

            
            return skewnewss,kurtosis

        # if values are continuous and have 0 as values
        if num:
            df[var]=df[var]+0.0001 ## remover 0
        
        #original var skewness and kurstosis
        positive_skewness=df[var].skew()
        print("Original Skew: ",positive_skewness)
        print("Original Kurtosis: ",df[var].kurt())

        if positive_skewness>0:
            print("Skewness is to the Left")

            # Log Transformation 
            transformed_data_log = np.log(df[var])
            log_skew=transformed_data_log.skew()
            log_kurt=transformed_data_log.kurt()
            # Sqrt Transformation 
            transformed_data_sqrt = np.sqrt(df[var])
            sqrt_skew=transformed_data_sqrt.skew()
            sqrt_kurt=transformed_data_sqrt.kurt()
            # Calculate metrics for each transformation
            if flag_prints:
                print("log_metric:")
                log_metric = metric(log_skew, log_kurt,flag_prints)
                print("log_metric: ",log_metric)
                print("sqrt_metric:")
                sqrt_metric = metric(sqrt_skew, sqrt_kurt,flag_prints)
                print("sqrt_metric: ",sqrt_metric)
            else:
                log_metric = metric(log_skew, log_kurt,flag_prints)
                sqrt_metric = metric(sqrt_skew, sqrt_kurt,flag_prints)
            # Choose the transformation with the smallest metric
            best_transformation = min([
                ('Log', abs(log_metric[0])+abs(log_metric[1]),transformed_data_log,log_skew, log_kurt),
                ('Sqrt', abs(sqrt_metric[0])+abs(sqrt_metric[1]),transformed_data_sqrt,sqrt_skew, sqrt_kurt)
            ], key=lambda x: x[1])
            return best_transformation
        
        else:
            print("Skewness is to the Right")
            # cube root Transformation 
            transformed_data_cube_root = np.cbrt(df[var])
            cube_root_skew=transformed_data_cube_root.skew()
            cube_root_kurt=transformed_data_cube_root.kurt()
            # Square Transformation 
            square_transformed_data = np.square(df[var])
            square_skew=square_transformed_data.skew()
            square_kurt=square_transformed_data.kurt()
            # Calculate metrics for each transformation
            if flag_prints:
                print("cube_root_metric:")
                cube_root_metric = metric(cube_root_skew, cube_root_kurt,flag_prints)
                print("cube_root_metric: ",cube_root_metric)
                print("square_metric:")
                square_metric = metric(square_skew,square_kurt,flag_prints)
                print("square_metric: ",square_metric)
            else:
                cube_root_metric = metric(cube_root_skew, cube_root_kurt,flag_prints)
                square_metric = metric(square_skew, square_kurt,flag_prints)
            # Choose the transformation with the smallest metric
            best_transformation = min([
                ('Cube_Root', abs(cube_root_metric[0])+abs(cube_root_metric[1]),transformed_data_cube_root,cube_root_skew, cube_root_kurt),
                ('Square', abs(square_metric[0])+abs(square_metric[1]),square_transformed_data,square_skew, square_kurt)
            ], key=lambda x: x[1])
            return best_transformation

    @staticmethod
    def transform_kurtosis(df,var,num=True,flag_prints=False): 
        return df