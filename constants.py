
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

experience_mapping2 = {
        '< 1 years': 0,
        '1-2 years': 0,
        '3-5 years': 1,
        '5-10 years': 1,
        '10-20 years': 2,
        '20+ years': 2,
        'I have never written code': -1,  # or any other appropriate value for missing or unspecified experience
    }

experience_mapping6 = {
        '$0 ($USD)': 0,
        '$1-$99': 0,
        '$100-$999': 1,
        '$1000-$9,999': 1,
        '$10,000-$99,999': 2,
        '$100,000 or more ($USD)': 2
    }

cat = ['Recommended pr. language', 'computing platform used', 'usage TPU',
           'Big Data Products', 'Primary Visualization tool', 'Q22']