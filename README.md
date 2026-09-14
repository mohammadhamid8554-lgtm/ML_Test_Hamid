# End to End Data Science Project

import dagshub
dagshub.init(repo_owner='mohammadhamid8554', repo_name='ML_Test_Hamid', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
