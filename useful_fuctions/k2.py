import pandas as pd
from sklearn.preprocessing import LabelEncoder


def shift(df:pd.DataFrame,columns:list[str],n:int) -> pd.DataFrame:
    df_copy = df.copy(deep=True)
    lags = range(n, 0, -1)
    for lag in lags:
        for column in columns:
            df_copy[f"{column}_{lag}"] = df_copy[column].shift(lag)
    return df_copy

# funkcija za bfill ffill i interpolate
def fill_missing_values(df:pd.DataFrame, column_name:str)->pd.DataFrame:

    df_copy = df.copy()
    first_valid_idx = df_copy[column_name].first_valid_index()
    last_valid_idx = df_copy[column_name].last_valid_index()

    df_copy.loc[:first_valid_idx, column_name] = df_copy.loc[:first_valid_idx, column_name].bfill()
    df_copy.loc[last_valid_idx:, column_name] = df_copy.loc[last_valid_idx:, column_name].ffill()
    df_copy[column_name] = df_copy[column_name].interpolate(method='linear')

    return df_copy

def qualitative_quantitative_attrs(df):
    qualitative_attributes = [col for col in df.columns if len(df[col].unique())<=10]
    quantitative_attributes = [col for col in df.columns if len(df[col].unique())>10]
    return qualitative_attributes,quantitative_attributes

def join_dataframes_by_index(df1, df2, how='inner'):
    return pd.merge(df1, df2, left_index=True, right_index=True, how=how)

#kastanje na integers
def cast_selected_columns(df: pd.DataFrame, columns: list, dtype: str) -> pd.DataFrame:
    df = df.copy()
    df[columns] = df[columns].astype(dtype)
    return df

#za timeseries lineplots
import matplotlib.pyplot as plt
import seaborn as sns
def plot_line_for_columns(df:pd.DataFrame, columns:list[str]):
    for col in columns:
        plt.figure(figsize=(5, 3))
        sns.lineplot(df[col])
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
def grouped_corr(df: pd.DataFrame):
    qualitative_attributes = [col for col in df.columns if len(df[col].unique()) <= 10]
    quantitative_attributes = [col for col in df.columns if len(df[col].unique()) > 10]
    for cat in qualitative_attributes:
        for num in quantitative_attributes:
            print(cat+"--->"+num)
            print("####")
            print(df.groupby(cat)[num].mean())
            print("####")
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=cat, y=num, data=df)
            plt.show()



def count_distributions(df:pd.DataFrame,columns:list[str]):
    for col in columns:
        plt.figure(figsize=(5,2))
        sns.countplot(x=col, data=df)
        plt.title(col)
        plt.show()


# df["Beat Annotation"] = df["Beat Annotation"].apply( lambda x : x if x in ['N','V'] else 'Other')

def encode(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_copy = dataframe.copy()
    for column in columns:
        le = LabelEncoder()
        mask = df_copy[column].isnull()
        df_copy[column] = le.fit_transform(df_copy[column].astype(str))
        df_copy.loc[mask, column] = None
    # or just ... labels = oe.fit_transform(df[["Value"]])
    return df_copy
"""
df['Stamp'] = pd.to_datetime(df['Stamp'], utc=True)
df['Stamp'] = df['Stamp'].dt.floor('15min')
pivot = df.pivot_table(index='Stamp', columns='Type', values='Value', aggfunc='mean')
pivot = pivot.resample('15min').first()

for i, column in enumerate(COLUMNS):
    print(f'{column} R2: {r2_score(y_test[column], y_pred[:, i])}')
    print(f'{column} RMSE: {np.sqrt(mean_squared_error(y_test[column], y_pred[:, i]))}')
    print('---' * 10)

pd.DataFrame({
    "terms": vectorizer.get_feature_names_out(),
    "coeffs": classificator.coef_[0],
}).nsmallest(10, "coeffs")

df['target'] = df['points'].apply(lambda x: 1 if x > 90 else 0)

dataset = Dataset.from_pandas(df[:500])
df = dataset["train"].to_pandas()
"""

"""
from datasets import load_dataset

dataset = load_dataset(
    'csv', 
    data_files='path/to/your_file.csv'
)

dataset = load_dataset(
    'csv',
    data_files='path/to/your_file.csv',
    split='train[:500]'
)

dataset = load_dataset(
    'csv', 
    data_files='path/to/your_file.tsv', 
    delimiter='\t'
)

dataset = load_dataset(
    'csv', 
    data_files=['file1.csv', 'file2.csv']
)


dataset = load_dataset(
    'csv', 
    data_files={
        'train': 'path/to/train.csv',
        'validation': 'path/to/val.csv',
        'test': 'path/to/test.csv'
    }
)

dataset = load_dataset(
    'csv', 
    data_files={
        'train': ['train_part1.csv', 'train_part2.csv'],
        'validation': 'val.csv'
    }
)
"""

#AGG
""" 
aggregation = {
    col: 'first' for col in df.columns[:-1]
}
aggregation.update({'News': '\n'.join})

rez:
            {'Open': 'first',
             'High': 'first',
             'Low': 'first',
             'Close': 'first',
             'Volume': 'first',
             'News': <function str.join(iterable, /)>}

df = df.groupby('Date').agg(aggregation)
"""
