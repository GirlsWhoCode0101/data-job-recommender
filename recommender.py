from imblearn.over_sampling import ADASYN
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

import constants as c


def add_data(data, column_name, percentage):
    data[column_name] = 0
    data_engineer_rows = data[data['Role'] == 'Data Engineer']
    num_ones = int(len(data_engineer_rows) * percentage)
    random_indices = np.random.choice(data_engineer_rows.index, size=num_ones, replace=False)
    data.loc[random_indices, column_name] = 1


@st.cache_data
def do_preprocessing(data):
    data.drop_duplicates(inplace=True)

    for col in c.to_be_reomved_cols:
        df_q = data.apply(lambda t: t.name.startswith(col))
        for index in range(data.shape[1]):
            if df_q[index]:
                data.drop(df_q.index[index], axis=1, inplace=True)

    data.drop(c.irrelevant_questions, axis=1, inplace=True)

    # drop first row in table which contains all the questions
    questions = data.iloc[0, 1:]
    data.drop(index=0, axis=0, inplace=True)

    data.rename(c.names_to_be_replaced, axis=1, inplace=True)

    data = data[data.Role.isin(
        ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Business Analyst'])]

    add_data(data, column_name='kafka', percentage=0.26)
    add_data(data, column_name='spark', percentage=0.5)
    add_data(data, column_name='hadoop', percentage=0.42)

    return questions, data


def replace_string(string):
    if isinstance(string, str):
        return 1
    else:
        return 0


class Recommender:
    def __init__(self, df):
        self.df_ = df
        self.features_ = df.drop('Role', axis=1)
        self.target_ = self.df_.Role
        self.feature_names_ = []
        self.lda_ = LDA()
        self.X_train_encoded_ = pd.DataFrame()
        self.X_LDA_valid_ = pd.DataFrame()
        self.svc_ = SVC(C=10, gamma=0.1, kernel='rbf', probability=True)

    def train_svc(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features_, self.target_, test_size=0.2,
                                                            random_state=42)

        X_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        for col in c.cat_binary:
            X_train[col] = X_train[col].apply(replace_string)

        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        cat_all = ['Recommended pr. language', 'Programming Experience',
                   'ML experience', 'Team spent on ML', 'computing platform used', 'usage TPU',
                   'Big Data Products', 'Primary Visualization tool', 'Q22']
        X_train[cat_all] = imputer.fit_transform(X_train[cat_all])

        # Convert the strings to numerical values
        X_train['ML experience'] = X_train['ML experience'].apply(lambda exp: c.experience_mapping.get(exp, -999))
        X_train['Programming Experience'] = X_train['Programming Experience'].apply(
            lambda exp: c.experience_mapping2.get(exp, -999))
        X_train['Team spent on ML'] = X_train['Team spent on ML'].apply(
            lambda exp: c.experience_mapping6.get(exp, -999))

        X_train_binary = X_train.drop(c.cat, axis=1)

        one = OneHotEncoder(drop='first', sparse=False)

        for col in c.cat:
            X_train[col] = X_train[col].map(str)

        # apply encoding to categorical non-order variables
        X_train_cat = one.fit_transform(X_train[c.cat])
        self.feature_names_ = one.get_feature_names_out(c.cat)
        X_train_cat = pd.DataFrame(X_train_cat, columns=self.feature_names_)

        # reset index
        X_train_binary = X_train_binary.reset_index(drop=True)
        X_train_cat = X_train_cat.reset_index(drop=True)
        self.X_train_encoded_ = pd.concat([X_train_binary, X_train_cat], axis=1)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train.map(str))

        adasyn = ADASYN()
        X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(self.X_train_encoded_, y_train)

        X_LDA = self.lda_.fit_transform(X_resampled_adasyn, y_resampled_adasyn)

        self.svc_.fit(X_LDA, y_resampled_adasyn)

    def do_encoding_for_prediction(self, X_valid):
        X_valid.reset_index(drop=True, inplace=True)

        for col in c.cat_binary:
            X_valid[col] = X_valid[col].apply(replace_string)
            X_valid[['hadoop', 'spark', 'kafka']] = X_valid[['hadoop', 'spark', 'kafka']].fillna(0)

        X_valid['ML experience'] = X_valid['ML experience'].apply(lambda exp: c.experience_mapping.get(exp, -999))
        X_valid['Programming Experience'] = X_valid['Programming Experience'].apply(
            lambda exp: c.experience_mapping2.get(exp, -999))
        X_valid['Team spent on ML'] = X_valid['Team spent on ML'].apply(
            lambda exp: c.experience_mapping6.get(exp, -999))

        for col in c.cat:
            X_valid[col] = X_valid[col].map(str)

        # do onehot encoding manually
        for col in range(0, len(c.cat)):
            column_name = X_valid[c.cat[col]].name + '_' + X_valid[c.cat[col]].values[0]
            X_valid[column_name] = 1
            X_valid.drop(X_valid[c.cat[col]].name, axis=1, inplace=True)

        missing_features = set(self.feature_names_) - set(X_valid.columns)
        for mf in missing_features:
            X_valid[mf] = 0

        # columns including options that have never been answered by people, are not
        # existent in the training set and therefore need to be removed from the test set
        column_diff = set(X_valid.columns) - set(self.X_train_encoded_.columns)
        if len(column_diff) != 0:
            X_valid.drop(column_diff, axis=1, inplace=True)

        # reorder x_test according to X_train
        new_column_order = self.X_train_encoded_.columns  # Get the desired column ordering from other_df
        X_valid = X_valid.reindex(columns=new_column_order)

        self.X_LDA_valid_ = self.lda_.transform(X_valid)
