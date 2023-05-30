import pandas as pd
from PIL import Image
import numpy as np
import recommender as rc
import re
import streamlit as st


def get_choice(q, df):
    df_sorted = df.iloc[:, q].unique()
    df_sorted = pd.Series(df_sorted)
    if len(df_sorted.dropna().values) == 1:
        return df_sorted.dropna().values[0]
    else:
        return df_sorted.dropna().values


pages = ['Submit your skills', 'Data Jobs Analysis']

st.sidebar.title('Info')
page = st.sidebar.radio("Do you like to know which data job suits you best? \n"
                        " Answer the questions on this page! \n", pages)

df = pd.read_csv('kaggle_survey_2020_responses.csv')
questions, df_preproc = rc.do_preprocessing(df)
recommender = rc.Recommender(df_preproc)
recommender.train_svc()

try:
    df_preproc.drop('Role', axis=1, inplace=True)
except KeyError:
    print('Already dropped column Role')
X_valid = pd.DataFrame(columns=df_preproc.columns)
# ToDo: clean questions series!!! do in a seperate function

for q in range(0, len(questions)):
    cleaned_question = re.findall(r'.*\(Select all that apply\)', questions[q])
    if len(cleaned_question) > 0:
        questions[q] = cleaned_question

if page == pages[0]:
    st.title("Data Job Recommender")
    st.header('Please answer the following questions:')

    questions_count = questions.value_counts()
    q_index = questions_count.index
    # first 13 questions are multiple selection options
    flat_index_question_list = [item for sublist in q_index[0:14].to_list() for item in sublist]
    not_used = [flat_index_question_list.append(item) for item in q_index[14:]]
    pattern = r"(?<=')[^']+(?=')|(?<=\")[^\"]+(?=\")"
    updated_question_index = 0

    for q in range(0, len(questions)):
        option_multi = []
        question = str(questions[q])
        try:
            index_question = flat_index_question_list.index(re.search(pattern, question).group(0))
        except AttributeError:
            index_question = flat_index_question_list.index(question)
        if questions_count[index_question] == 1:
            choice = get_choice(q, df_preproc)
            option = st.selectbox(questions[q], tuple(choice))
            X_valid.loc[0, df_preproc.iloc[:, q].name] = option
        else:
            # check if we already elaborated on this question
            if q >= updated_question_index:
                choice = []
                for cnt in range(q, questions_count[index_question] + q):
                    choice.append(get_choice(cnt, df_preproc))
                choice = list(map(lambda obj: 'None' if isinstance(obj, (np.ndarray, np.generic)) else obj, choice))
                updated_question_index = q + questions_count[index_question]
                option_multi = st.multiselect(questions[q][0], choice, default=None)
        if len(option_multi) > 0:
            for opt in option_multi:
                for c in range(q, updated_question_index):
                    if df_preproc.iloc[:, c].name == opt:
                        X_valid.loc[0, df_preproc.iloc[:, c].name] = opt

    option = st.multiselect('Do you have experience with Hadoop, Spark or Kafka? Select all that apply.',
                        ['spark', 'hadoop', 'kafka', 'None'],
                        default=None)
    for opt in option:
        for c in range(-3, 0):
            if df_preproc.iloc[:, c].name == opt:
                X_valid.loc[0, df_preproc.iloc[:, c].name] = 1

    button_clicked = st.button('Submit your answers.')

    if button_clicked:
        recommender.do_encoding_for_prediction(X_valid)
        probabilities_class = recommender.svc_.predict_proba(recommender.X_LDA_valid_)
        print(probabilities_class)
        index_max_class = probabilities_class.argmax()
        classes = {0: 'Business Analyst', 1: 'Data Analyst', 2: 'Data Engineer', 3: 'Data Scientest', 4: 'ML Engineer'}

        if probabilities_class.max() > 0.6:
            st.write('With your skill set, we recommend a role as: ', classes[index_max_class])
        else:
            st.write('You should put more effort in your education and skill set to fit in the Data World.')

if page == pages[1]:
    image = Image.open('DataJob_dash.png')
    st.image(image, caption='Data Job Survey 2020 insights', use_column_width=True)