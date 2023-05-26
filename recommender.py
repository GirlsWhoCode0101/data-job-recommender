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


def replace_none(series):
    if series[0] is None:
        return 0


def recommender_svc(df, target, X_valid):
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    # for applying recommender system to peoples choices on questionare
    X_test = X_valid

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    cat_binary = ['Python', 'R', 'SQL', 'C', 'C++', 'Java',
                  'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB used',
                  'None pr. language', 'Other pr.languages',
                  'JupyterLab', 'RStudio', 'Visual Studio', 'Visual Studio Code',
                  'PyCharm', 'Spyder', 'Notepad++', 'Sublime Text', 'Vim / Emacs',
                  'MATLAB', 'No IDE', 'Q9_OTHER', 'Kaggle Notebooks', 'Colab Notebooks',
                  'Azure Notebooks', 'Paperspace / Gradient', 'Binder / JupyterHub',
                  'Code Ocean', 'IBM Watson Studio', 'Amazon Sagemaker Studio',
                  'Amazon EMR Notebooks', 'Google Cloud AI Platform Notebooks',
                  'Google Cloud Datalab Notebooks', 'Databricks Collaborative Notebooks',
                  'No Notebook', 'Other Notebook', 'GPUs',
                  'TPUs', 'No HW', 'other HW', 'Matplotlib', 'Seaborn',
                  'Plotly / Plotly Express', 'Ggplot / ggplot2', 'Shiny', 'D3js',
                  'Altair', 'Bokeh', 'Geoplotlib', 'Leaflet / Folium', 'No libs',
                  'Q14_OTHER', 'Scikit-learn',
                  'Decision Trees or Random Forests', 'Keras', 'PyTorch', 'Fast.ai',
                  'MXNet', 'Xgboost', 'LightGBM', 'CatBoost', 'Prophet', 'H2O3', 'Caret',
                  'Tidymodels', 'JAX', 'No ML framework used', 'Q16_OTHER', 'Linear or Logistic Regression',
                  'TensorFlow', 'Gradient Boosting Machines', 'Bayesian Approaches',
                  'Evolutionary Approaches', 'Dense Neural Networks',
                  'Convolutional Neural Networks', 'Generative Adversarial Networks',
                  'Recurrent Neural Networks', 'Transformer Networks', 'No ML algorithm',
                  'Q17_OTHER', 'Q23_Part_1', 'Q23_Part_2',
                  'Q23_Part_3', 'Q23_Part_4', 'Q23_Part_5', 'Q23_Part_6', 'Q23_Part_7',
                  'Q23_OTHER',
                  'Amazon Web Services', 'Microsoft Azure', 'Google Cloud Platform',
                  'Q26_A_Part_4', 'Q26_A_Part_5', 'Q26_A_Part_6', 'Q26_A_Part_7',
                  'Q26_A_Part_8', 'Q26_A_Part_9', 'Q26_A_Part_10', 'No cloud pl. used',
                  'Q26_A_OTHER', 'Amazon EC2', 'AWS Lambda',
                  'Amazon Elastic Container Service', 'Azure Cloud Services',
                  'Microsoft Azure Container Instances', 'Azure Functions',
                  'Google Cloud Compute Engine', 'Google Cloud Functions',
                  'Google Cloud Run', 'Google Cloud App Engine', 'No cloud c. platform',
                  'Other cloud c. platform', 'MySQL', 'PostgresSQL', 'SQLite', 'Q29_A_Part_4',
                  'MongoDB', 'Q29_A_Part_6', 'Q29_A_Part_7',
                  'Microsoft SQL Server', 'Q29_A_Part_9', 'Q29_A_Part_10',
                  'Q29_A_Part_11', 'Q29_A_Part_12', 'Q29_A_Part_13', 'Q29_A_Part_14',
                  'Q29_A_Part_15', 'Q29_A_Part_16', 'No big data', 'Q29_A_OTHER',
                  'Q36_Part_1', 'Q36_Part_2', 'Q36_Part_3',
                  'Q36_Part_4', 'Q36_Part_5', 'Q36_Part_6', 'Q36_Part_7', 'Q36_Part_8',
                  'Q36_Part_9', 'Q36_OTHER', 'Q37_Part_1', 'Q37_Part_2', 'Q37_Part_3',
                  'Q37_Part_4', 'Q37_Part_5', 'Q37_Part_6', 'Q37_Part_7', 'Q37_Part_8',
                  'Q37_Part_9', 'Q37_Part_10', 'Q37_Part_11', 'Q37_OTHER',
                  'Q39_Part_1', 'Q39_Part_2', 'Q39_Part_3', 'Q39_Part_4', 'Q39_Part_5',
                  'Q39_Part_6', 'Q39_Part_7', 'Q39_Part_8', 'Q39_Part_9', 'Q39_Part_10',
                  'Q39_Part_11', 'Q39_OTHER']

    for col in cat_binary:
        X_train[col] = X_train[col].apply(replace_string)
        X_test[col] = X_test[col].apply(replace_string)

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    cat = ['Recommended pr. language', 'Programming Experience',
           'ML experience', 'Team spent on ML', 'computing platform used', 'usage TPU',
           'Big Data Products', 'Primary Visualization tool', 'Q22']

    X_train[cat] = imputer.fit_transform(X_train[cat])
    # X_test[cat] = X_test[cat].apply(replace_none)
    X_test[['hadoop', 'spark', 'kafka']] = X_test[['hadoop', 'spark', 'kafka']].fillna(0)

    experience_mapping = {
        'Under 1 year': 0,
        '1-2 years': 0,
        '2-3 years': 0,
        '3-4 years': 1,
        '4-5 years': 1,
        '5-10 years': 2,
        '10-20 years': 3,
        '20 or more years': 3,
        'I do not use machine learning methods': -1,
    }

    # Convert the strings to numerical values
    X_train['ML experience'] = X_train['ML experience'].apply(lambda exp: experience_mapping.get(exp, -999))
    X_test['ML experience'] = X_test['ML experience'].apply(lambda exp: experience_mapping.get(exp, -999))

    experience_mapping2 = {
        '< 1 years': 0,
        '1-2 years': 0,
        '3-5 years': 1,
        '5-10 years': 1,
        '10-20 years': 2,
        '20+ years': 2,
        'I have never written code': -1,  # or any other appropriate value for missing or unspecified experience
    }

    X_train['Programming Experience'] = X_train['Programming Experience'].apply(
        lambda exp: experience_mapping2.get(exp, -999))
    X_test['Programming Experience'] = X_test['Programming Experience'].apply(
        lambda exp: experience_mapping2.get(exp, -999))

    experience_mapping6 = {
        '$0 ($USD)': 0,
        '$1-$99': 0,
        '$100-$999': 1,
        '$1000-$9,999': 1,
        '$10,000-$99,999': 2,
        '$100,000 or more ($USD)': 2
    }

    X_train['Team spent on ML'] = X_train['Team spent on ML'].apply(lambda exp: experience_mapping6.get(exp, -999))
    X_test['Team spent on ML'] = X_test['Team spent on ML'].apply(lambda exp: experience_mapping6.get(exp, -999))

    cat = ['Recommended pr. language', 'computing platform used', 'usage TPU',
           'Big Data Products', 'Primary Visualization tool', 'Q22']
    X_train_binary = X_train.drop(cat, axis=1)

    one = OneHotEncoder(drop='first', sparse=False)

    for col in cat:
        X_train[col] = X_train[col].map(str)
        X_test[col] = X_test[col].map(str)

    # for to predicted X-test add columns from onehot encoding manually
    for col in range(0, len(cat)):
        column_name = X_test[cat[col]].name + '_' + X_test[cat[col]].values[0]
        X_test[column_name] = 1
        X_test.drop(X_test[cat[col]].name, axis=1, inplace=True)

    # apply encoding to categorical non-order variables
    X_train_cat = one.fit_transform(X_train[cat])
    feature_names = one.get_feature_names_out(cat)
    X_train_cat = pd.DataFrame(X_train_cat, columns=feature_names)

    # compare 2 sets of feature names from X-train and X-test and add missing ones
    # X-test with value of 0
    missing_features = set(feature_names) - set(X_test.columns)
    # ToDo. list comprehension or map
    for mf in missing_features:
        X_test[mf] = 0

    # reset index
    X_train_binary = X_train_binary.reset_index(drop=True)
    X_train_cat = X_train_cat.reset_index(drop=True)
    X_train_encoded = pd.concat([X_train_binary, X_train_cat], axis=1)

    X_test_encoded = X_test

    # columns including options that have never been answered by people, are not
    # existent in the training set and therefore need to be removed from the test set
    column_diff = set(X_test_encoded.columns) - set(X_train_encoded.columns)
    if len(column_diff) != 0:
        X_test_encoded.drop(column_diff, axis=1, inplace=True)

    # reorder x_test according to X_train
    new_column_order = X_train_encoded.columns  # Get the desired column ordering from other_df
    X_test_encoded = X_test_encoded.reindex(columns=new_column_order)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train.map(str))

    adasyn = ADASYN()
    X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train_encoded, y_train)

    lda = LDA()
    X_LDA = lda.fit_transform(X_resampled_adasyn, y_resampled_adasyn)
    X_LDA_test = lda.transform(X_test_encoded)

    # Classifier
    y_train = y_resampled_adasyn
    X_train = X_LDA
    svc = SVC(C=10, gamma=0.1, kernel='rbf', probability=True)
    svc.fit(X_train, y_train)

    #y_pred = svc.predict(X_test)
    #y_pred_subset = y_pred[:len(y_test)]
    #print(pd.crosstab(y_test, y_pred_subset, rownames=['True'], colnames=['Prediction']))
    #print(classification_report(y_test, y_pred_subset, ))
    #accuracy = accuracy_score(y_test, y_pred_subset)
    #print("Accuracy of the SVC model:", accuracy)

    return svc, X_LDA_test