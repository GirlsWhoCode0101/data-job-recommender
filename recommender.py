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


@st.cache_data
def do_preprocessing(data):
    data.drop_duplicates(inplace=True)
    col = ['Time from Start to Finish (seconds)', 'Q18', 'Q19', 'Q28_A', 'Q32', 'Q34_A', 'Q26_B', 'Q27_B', 'Q28_B',
           'Q29_B', 'Q30_B', 'Q31_A', 'Q31_B', 'Q32_B', 'Q33_A', 'Q33_B', 'Q34_B', 'Q35_A', 'Q35_B']

    for c in col:
        df_q = data.apply(lambda t: t.name.startswith(c))
        for index in range(data.shape[1]):
            if df_q[index]:
                data.drop(df_q.index[index], axis=1, inplace=True)

    data.drop('Q3', axis=1, inplace=True)
    data.drop('Q2', axis=1, inplace=True)
    data.drop('Q1', axis=1, inplace=True)
    data.drop('Q4', axis=1, inplace=True)
    data.drop('Q21', axis=1, inplace=True)
    data.drop('Q20', axis=1, inplace=True)
    data.drop('Q24', axis=1, inplace=True)

    # drop first row in table which contains all the questions
    questions = data.iloc[0, 1:]
    data.drop(index=0, axis=0, inplace=True)

    data.rename({'Q5': 'Role',
                 'Q6': 'Programming Experience', 'Q7_Part_1': 'Python', 'Q7_Part_2': 'R', 'Q7_Part_3': 'SQL',
                 'Q7_Part_4': 'C', 'Q7_Part_5': 'C++',
                 'Q7_Part_6': 'Java', 'Q7_Part_7': 'Javascript', 'Q7_Part_8': 'Julia', 'Q7_Part_9': 'Swift',
                 'Q7_Part_10': 'Bash', 'Q7_Part_11': 'MATLAB used', 'Q7_Part_12': 'None pr. language',
                 'Q7_OTHER': 'Other pr.languages',
                 'Q8': 'Recommended pr. language', 'Q9_Part_1': 'JupyterLab', 'Q9_Part_2': 'RStudio',
                 'Q9_Part_3': 'Visual Studio', 'Q9_Part_4': 'Visual Studio Code', 'Q9_Part_5': 'PyCharm',
                 'Q9_Part_6': 'Spyder',
                 'Q9_Part_7': 'Notepad++', 'Q9_Part_8': 'Sublime Text', 'Q9_Part_9': 'Vim / Emacs',
                 'Q9_Part_10': 'MATLAB', 'Q9_Part_11': 'No IDE', 'Q9_Part_12': 'Other IDE',
                 'Q10_Part_1': 'Kaggle Notebooks', 'Q10_Part_2': 'Colab Notebooks', 'Q10_Part_3': 'Azure Notebooks',
                 'Q10_Part_4': 'Paperspace / Gradient', 'Q10_Part_5': 'Binder / JupyterHub', 'Q10_Part_6': 'Code Ocean',
                 'Q10_Part_7': 'IBM Watson Studio', 'Q10_Part_8': 'Amazon Sagemaker Studio',
                 'Q10_Part_9': 'Amazon EMR Notebooks',
                 'Q10_Part_10': 'Google Cloud AI Platform Notebooks', 'Q10_Part_11': 'Google Cloud Datalab Notebooks',
                 'Q10_Part_12': 'Databricks Collaborative Notebooks', 'Q10_Part_13': 'No Notebook',
                 'Q10_OTHER': 'Other Notebook',
                 'Q11': 'computing platform used',
                 'Q12_Part_1': 'GPUs', 'Q12_Part_2': 'TPUs', 'Q12_Part_3': 'No HW', 'Q12_OTHER': 'other HW',
                 'Q13': 'usage TPU',
                 'Q14_Part_1': 'Matplotlib', 'Q14_Part_2': 'Seaborn', 'Q14_Part_3': 'Plotly / Plotly Express',
                 'Q14_Part_4': 'Ggplot / ggplot2', 'Q14_Part_5': 'Shiny', 'Q14_Part_6': 'D3js', 'Q14_Part_7': 'Altair',
                 'Q14_Part_8': 'Bokeh', 'Q14_Part_9': 'Geoplotlib', 'Q14_Part_10': 'Leaflet / Folium',
                 'Q14_Part_11': 'No libs',
                 'Q14_Part_12': 'Other libs', 'Q15': 'ML experience', 'Q16_Part_1': 'Scikit-learn',
                 'Q16_Part_2': 'Decision Trees or Random Forests',
                 'Q16_Part_3': 'Keras', 'Q16_Part_4': 'PyTorch', 'Q16_Part_5': 'Fast.ai', 'Q16_Part_6': 'MXNet',
                 'Q16_Part_7': 'Xgboost',
                 'Q16_Part_8': 'LightGBM', 'Q16_Part_9': 'CatBoost', 'Q16_Part_10': 'Prophet', 'Q16_Part_11': 'H2O3',
                 'Q16_Part_12': 'Caret', 'Q16_Part_13': 'Tidymodels', 'Q16_Part_14': 'JAX',
                 'Q16_Part_15': 'No ML framework used',
                 'Q16_Part_16': 'Other ML framework used', 'Q17_Part_1': 'Linear or Logistic Regression',
                 'Q17_Part_2': 'TensorFlow',
                 'Q17_Part_3': 'Gradient Boosting Machines', 'Q17_Part_4': 'Bayesian Approaches',
                 'Q17_Part_5': 'Evolutionary Approaches',
                 'Q17_Part_6': 'Dense Neural Networks', 'Q17_Part_7': 'Convolutional Neural Networks',
                 'Q17_Part_8': 'Generative Adversarial Networks', 'Q17_Part_9': 'Recurrent Neural Networks',
                 'Q17_Part_10': 'Transformer Networks',
                 'Q17_Part_11': 'No ML algorithm', 'Q17_Part_12': 'Other ML algorithm',
                 'Q25': 'Team spent on ML', 'Q26_A_Part_1': 'Amazon Web Services',
                 'Q26_A_Part_2': 'Microsoft Azure', 'Q26_A_Part_3': 'Google Cloud Platform',
                 'Q26_A_Part_11': 'No cloud pl. used',
                 'Q27_A_Part_1': 'Amazon EC2', 'Q27_A_Part_2': 'AWS Lambda',
                 'Q27_A_Part_3': 'Amazon Elastic Container Service',
                 'Q27_A_Part_4': 'Azure Cloud Services', 'Q27_A_Part_5': 'Microsoft Azure Container Instances',
                 'Q27_A_Part_6': 'Azure Functions', 'Q27_A_Part_7': 'Google Cloud Compute Engine',
                 'Q27_A_Part_8': 'Google Cloud Functions',
                 'Q27_A_Part_9': 'Google Cloud Run', 'Q27_A_Part_10': 'Google Cloud App Engine',
                 'Q27_A_Part_11': 'No cloud c. platform', 'Q27_A_OTHER': 'Other cloud c. platform',
                 'Q29_A_Part_1': 'MySQL', 'Q29_A_Part_2': 'PostgresSQL', 'Q29_A_Part_3': 'SQLite',
                 'Q29_A_Part_5': 'MongoDB',
                 'Q29_A_Part_8': 'Microsoft SQL Server', 'Q29_A_Part_17': 'No big data', 'Q30': 'Big Data Products',
                 'Q38': 'Primary Visualization tool'}, axis=1, inplace=True)

    data = data[data.Role.isin(
        ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Business Analyst'])]

    data['kafka'] = 0
    data_engineer_rows = data[data['Role'] == 'Data Engineer']
    num_ones = int(len(data_engineer_rows) * 0.26)
    random_indices = np.random.choice(data_engineer_rows.index, size=num_ones, replace=False)
    data.loc[random_indices, 'kafka'] = 1

    data['spark'] = 0
    data_engineer_rows = data[data['Role'] == 'Data Engineer']
    num_ones = int(len(data_engineer_rows) * 0.5)
    random_indices = np.random.choice(data_engineer_rows.index, size=num_ones, replace=False)
    data.loc[random_indices, 'spark'] = 1

    data['hadoop'] = 0
    data_engineer_rows = data[data['Role'] == 'Data Engineer']
    num_ones = int(len(data_engineer_rows) * 0.42)
    random_indices = np.random.choice(data_engineer_rows.index, size=num_ones, replace=False)
    data.loc[random_indices, 'hadoop'] = 1

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
        X_train, X_test, y_train, y_test = train_test_split(self.features_, self.target_, test_size=0.2, random_state=42)

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
        X_train['Team spent on ML'] = X_train['Team spent on ML'].apply(lambda exp: c.experience_mapping6.get(exp, -999))

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
        X_valid['Team spent on ML'] = X_valid['Team spent on ML'].apply(lambda exp: c.experience_mapping6.get(exp, -999))

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