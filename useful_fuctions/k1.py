from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def impute_knn(df:pd.DataFrame, columns:list, n_neighbors=5):
    df_copy = df.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_copy[columns])
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = knn_imputer.fit_transform(scaled_data)
    df_copy[columns] = imputed_data
    return df_copy



def label_encoder(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_copy = dataframe.copy()
    for column in columns:
        le = LabelEncoder()
        mask = df_copy[column].isnull()
        df_copy[column] = le.fit_transform(df_copy[column].astype(str))
        df_copy.loc[mask, column] = None
    # or just ... labels = oe.fit_transform(df[["Value"]])
    return df_copy


def simple_imputer(dataframe: pd.DataFrame, strategy: str, columns) -> pd.DataFrame:
    df_copy = dataframe.copy()
    for column_name in columns:
        imputer = SimpleImputer(strategy=strategy)
        df_copy[column_name] = imputer.fit_transform(df_copy[[column_name]])
    return df_copy

def grouped_corr(df: pd.DataFrame):
    for cat in df.select_dtypes(include=['object', 'category']):
        for num in df.select_dtypes(include=['number']):
            print(cat+"--->"+num)
            print("####")
            print(df.groupby(cat)[num].mean())
            print("####")


def not_important(df: pd.DataFrame,target:str):
    not_important_attributes = list(df.columns)
    not_important_attributes.remove(target)
    thresh = 0.1

    for i in df.columns:
        for j in df.columns:
            if i == j:
                continue
            if df[i].corr(df[j]) > thresh or df[i].corr(df[j]) < -thresh:
                if i in not_important_attributes:
                    not_important_attributes.remove(i)
                break

    return not_important_attributes

def qualitative_quantitative_attrs(df):
    qualitative_attributes = [col for col in df.columns if len(df[col].unique())<10]
    quantitative_attributes = [col for col in df.columns if len(df[col].unique())>10]
    return qualitative_attributes,quantitative_attributes

#casting to int
def cast_to_int(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    df[columns] = df[columns].astype(int)
    return df