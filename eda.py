# This study has been prepared in order to practice exploratory data analysis, function definition and document creation together.
# Within the scope of the study, the functions load_df, check_df, grab_col_names and cat_summary are defined.
# It can run functions by choosing any of the dataset names in the sns library.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# In order not to rewrite the same code, we define the function where we will do the read dataframe.
def load_df(df_name):
    df = sns.load_dataset(df_name)
    return df

#Define a function for categorical and numerical variable analysis.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables include numeric looking categorical variables.

    Parameters
    ----------
    dataframe: dataframe
        The df for which variable names are to be found
    cat_th: int, optional
        frequency threshold for numeric but categorical variables
    car_th: int, optional
        frequency threshold for categorical but cardinal variables

    Returns
    -------
        cat_cols: list
            Categorical variable list
        num_cols: list
            Numeric variable list
        cat_but_car: list
            Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if (dataframe[col].nunique() < cat_th) and (dataframe[col].dtypes != "O")]

    cat_but_car = [col for col in dataframe.columns if (dataframe[col].nunique() > car_th) and (dataframe[col].dtypes == "O")]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# We define a function to perform exploratory data analysis on the dataframe.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# We define a function to analyze the frequency value of a categorical variable and its ratio in the total.
def cat_summary(dataframe, col_name, plot=False, ratio_type=True, float_digit=2):
    if ratio_type:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": ((100 * dataframe[col_name].value_counts() / len(dataframe))).astype(int)}))
        print("##########################################")
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": round(100 * dataframe[col_name].value_counts() / len(dataframe),float_digit)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

df = load_df("diamonds", ".csv")

check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_summary(df,"cut",plot=False, ratio_type=False, float_digit=2)
cat_summary(df,"color",plot=False, ratio_type=True)
