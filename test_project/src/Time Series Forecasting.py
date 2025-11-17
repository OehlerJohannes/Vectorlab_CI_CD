# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Forecasting Model on Databricks

# COMMAND ----------

#Install Dependencies
!pip install prophet
!pip install databricks-sdk --upgrade
!pip install mlflow
!pip install grpcio
!pip install grpcio-status
!pip install pandas
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Selection
# MAGIC Load your training data from a delta table and split it into training and testing data.

# COMMAND ----------

# TODO: update the catalog, schema, and table name for your data and give your model a name
catalog = "johannes_oehler"
schema = "vectorlab"
table = "forecast_data"
forecast_horizon = 10


# Define the catalog, schema, and model name for organizing the model within the MLflow model registry
model_catalog = "johannes_oehler" #Update it to your catalog name
model_schema = "vectorlab" #Update it to your schema name
model_name = "sales_forecast" #Update it to your model name

serving_endpoint_name = "forecast_joe"

# COMMAND ----------

#Select Data
query = f"SELECT date, store, SUM(sales) as sales FROM {catalog}.{schema}.{table} GROUP BY date, store ORDER BY date desc"

df = spark.sql(query)

# Choose a single store to make the calculations simpler
df = df.filter(df.store == 1)

# train-test-split
train_df = df.orderBy(df.date.asc()).limit(df.count() - forecast_horizon).orderBy(df.date.desc())
test_df = df.orderBy(df.date.desc()).limit(forecast_horizon).toPandas()

train_df.show(5)
test_df.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC Effectively preparing your data is a foundational step in the forecasting process. Proper preparation ensures the accuracy and reliability of your model's predictions. 
# MAGIC
# MAGIC Data preparation steps can include:
# MAGIC - Missing Values: Identify and address missing values in your data. Common strategies include deletion (if minimal), imputation (filling in missing values with statistical methods or previous observations), or interpolation (estimating missing values based on surrounding data points).
# MAGIC - Outliers: Identify and handle outliers, which are extreme data points that can significantly distort your forecasts. You can choose to remove outliers if they are truly erroneous or winsorize them (capping their values to a certain threshold).
# MAGIC - Time Consistency: Ensure your data has consistent timestamps and that the time steps are evenly spaced (e.g., daily data points should be recorded at the same time each day).
# MAGIC Feature Engineering: Create new features from existing ones if it can improve the forecasting model's performance. This might involve calculating rolling averages, seasonality indicators, or lag features (past values of the target variable).

# COMMAND ----------

from pyspark.sql.functions import col, lit

# Dropping rows with missing values in the 'sales' column
cleaned_df = train_df.na.drop(subset=["sales"]) 
cleaned_df.show(5)

# Calculating IQR and defining bounds for outliers
quartiles = cleaned_df.approxQuantile("sales", [0.25, 0.75], 0.05) 
IQR = quartiles[1] - quartiles[0]
lower_bound = 0
upper_bound = quartiles[1] + 1.5 * IQR

# Filtering out outliers
no_outliers_df = cleaned_df.filter(
    (col("sales") > lit(lower_bound)) 
    & (col("sales") <= lit(upper_bound)) 
)

# Showing the updated DataFrame
no_outliers_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Selection
# MAGIC The model you choose will depend on the nature of your data and the specific forecasting problem you’re trying to solve. Take into consideration different data characteristics such as Frequency (daily vs weekly), granularity (hourly vs daily sales), seasonality, and any other external factors such as holidays, promotions etc. 
# MAGIC
# MAGIC Machine learning methods:
# MAGIC
# MAGIC - Prophet: User-friendly and specifically designed for time series forecasting, offering built-in seasonality and holiday handling.
# MAGIC - ARIMA: A classical statistical method for time series forecasting, capturing short-term patterns and trends in stationary data through autocorrelation, differencing, and moving average components.
# MAGIC - LSTMs (Long Short-Term Memory): Powerful for capturing complex relationships and long-term dependencies in time series data.
# MAGIC For the scope of this tutorial, Prophet will serve as our primary model. Prophet stands out for its user-friendly nature and robust handling of various time series forecasting challenges, making it a versatile option for a wide range of applications.

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook includes the training of a Prophet model
# MAGIC
# MAGIC **TODO: train and test additional models and examine, which model performs best**

# COMMAND ----------

from prophet import Prophet
from pyspark.sql.functions import col, to_date

# Prophet requires at the minimum 2 columns - ds & y
train_df = no_outliers_df.select(to_date(col("date")).alias("ds"), col("store"), col("sales").alias("y").cast("double")).orderBy(col("ds").desc())

# set model parameters
prophet_model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=True,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='additive'
  )
 
# fit the model to historical data
history_pd = train_df.toPandas()
prophet_model.fit(history_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC # Fit Data and Build Forecast
# MAGIC After fitting the model, you will create a DataFrame that outlines the future dates you wish to predict. Using this DataFrame, Prophet will generate forecasts for the specified future dates. 

# COMMAND ----------


#Define Dataset with historical dates & x-days beyond the last available date
future_pd = prophet_model.make_future_dataframe(
  periods=forecast_horizon, 
  freq='d', 
  include_history=True
  )
 
#Forecast
forecast_pd = prophet_model.predict(future_pd)
display(forecast_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluation
# MAGIC This can be done by comparing the model’s forecasts to actual data and calculating performance metrics like Mean Squared Error (MSE). Performance Metrics:
# MAGIC
# MAGIC - Mean Squared Error (MSE): Measures the average squared difference between the estimated values and the actual value, offering a view of the overall variance in the forecasting errors. Lower MSE values denote a model with fewer errors.
# MAGIC - Root Mean Squared Error (RMSE): Represents the square root of MSE, thus re-scaling errors to the original units of the target variable, which improves interpretability.
# MAGIC - Mean Absolute Error (MAE): Averages the absolute differences between predicted and actual values. Unlike MSE, MAE is more robust to outliers, as it does not square the errors.
# MAGIC
# MAGIC Interpreting the Metrics
# MAGIC
# MAGIC MSE and RMSE are more sensitive to outliers due to squaring the errors, often used when large errors are particularly undesirable.
# MAGIC MAE is straightforward and easy to interpret, as it directly represents the average error.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO: Evaluate your own models**

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO: Use additional evaluation metrics**

# COMMAND ----------

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from datetime import date

# get historical actuals & predictions for comparison
actuals_pd = test_df['sales']
predicted_pd = forecast_pd[forecast_pd['ds'] >= pd.to_datetime(test_df['date'].min())]['yhat'] #Update it to max date on your dataset

# calculate evaluation metrics
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(actuals_pd, predicted_pd)

# Print other metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

# TODO: add more metrics

# COMMAND ----------

# TODO: evaluate your own models

# COMMAND ----------

# MAGIC %md
# MAGIC # Log the model with MLflow
# MAGIC Model Logging is critical for tracking the performance and changes to models over time, ensuring reproducibility and accountability. Techniques include:
# MAGIC
# MAGIC - MLflow Logging: Utilize MLflow's robust platform for logging models, parameters, and artifacts. It supports structured experiment tracking, perfect for recording and comparing different versions of your models.
# MAGIC - Custom Logging: Implement tailored logging approaches to capture unique model insights or additional metadata not standardly logged by existing tools.
# MAGIC
# MAGIC ## Benefits of Logging with MLflow
# MAGIC Logging models with MLflow offers several advantages:
# MAGIC
# MAGIC - Reproducibility: By capturing all necessary details of the experimentation phase, MLflow makes it easier to replicate results and understand decision-making processes.
# MAGIC - Model Registry: MLflow allows for versioning of models, making it simple to manage and deploy specific model versions based on performance metrics.
# MAGIC - Collaboration and Sharing: Teams can leverage MLflow’s centralized model storage to share models and results, enhancing collaboration.
# MAGIC For the purpose of this tutorial, we demonstrate how to efficiently log a Prophet model using MLflow, capturing essential information that supports further analysis and model deployment.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO: log your own model**

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel, log_model


class MyPythonModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        future_pd = self.model.make_future_dataframe(periods=10, freq="d", include_history=True)
        forecast_pd = self.model.predict(future_pd)
        return forecast_pd[["ds", "yhat", "yhat_upper", "yhat_lower"]]
    
wrapped_model = MyPythonModel(prophet_model)

# Enable MLflow auto logging for tracking machine learning metrics and artifacts
with mlflow.start_run(run_name="Prophet Model Run") as run:

    input_example = history_pd.head()[["ds", "y"]]
    output_example = prophet_model.predict(input_example).iloc[:10]

    # Log calculated metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    print(output_example)
    # Infer the signature of the machine learning model
    signature = infer_signature(input_example, output_example)

    # Update dependencies in the default conda environment
    env = mlflow.pyfunc.get_default_conda_env()
    env['dependencies'][-1]['pip'] = [pkg for pkg in env['dependencies'][-1]['pip'] if ("cloudpickle" not in pkg) and ("mlflow" not in pkg)]
    env['dependencies'][-1]['pip'] += ["prophet==1.1.6"]
    env['dependencies'][-1]['pip'] += ["pandas==1.5.3"]
    env['dependencies'][-1]['pip'] += ["pyspark==4.0.0"]
    env['dependencies'][-1]['pip'] += ["grpcio==1.69.0"]
    env['dependencies'][-1]['pip'] += ["grpcio_status==1.69.0"]
    env['dependencies'][-1]['pip'] += ["numpy==1.23.5"] #1.26.4
    env['dependencies'][-1]['pip'] += ["mlflow==2.22.0"] 
    env['dependencies'][-1]['pip'] += ["cloudpickle==3.0.0"] 
    
    # Log the trained model to MLflow with the inferred signature
    model_log = log_model(
        artifact_path="forecasting_model",
        python_model=wrapped_model,
        signature=signature,
        input_example=input_example,
        registered_model_name=f"{model_catalog}.{model_schema}.{model_name}",
        conda_env=env
    )

    # Retain the "run_id" for use with other MLflow functionalities like registering the model
    run_id = run.info.run_id


# COMMAND ----------

 input_example = history_pd.head()[["ds", "y"]]

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy Models on Databricks
# MAGIC Deploying machine learning models into production on Databricks can be achieved through two primary methods: MLflow for batch inference and prediction, and Databricks Model Serving for real-time inference. Each serves different use cases based on the requirement for real-time responses and the scale of data processing.
# MAGIC
# MAGIC - MLflow for Batch Inference and Prediction: Batch processing is ideal for scenarios where predictions can be made on large datasets at once without the need for immediate responses. This method fits well with scheduled analytics and reporting.
# MAGIC - Databricks Model Serving for Real-Time Inference: This method is better suited for scenarios where low latency and real-time responses are important.

# COMMAND ----------

#Get the latest Model Version
def get_latest_model_version(model_name:str = None):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version
  
model_version = get_latest_model_version(f"{model_catalog}.{model_schema}.{model_name}")
model_version

# COMMAND ----------

import mlflow, os
import requests, json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from mlflow.deployments import get_deploy_client


# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)


client = get_deploy_client("databricks")

# Check if the endpoint already exists
existing_endpoint = next(
    (e for e in client.list_endpoints() if e['name'] == serving_endpoint_name), None
)

# Update the endpoint configuration
endpoint_config = {
    "served_entities": [
        {
            "entity_name": f"{model_catalog}.{model_schema}.{model_name}",
            "entity_version": model_version,
            "workload_size": "Small",
            "workload_type": "CPU",
            "scale_to_zero_enabled": True
        }
    ]
}

if existing_endpoint is not None:
    # Update the existing endpoint
    endpoint = client.update_endpoint(
        endpoint=serving_endpoint_name,
        config=endpoint_config
    )
else:
    # Create a new endpoint if it does not exist
    endpoint = client.create_endpoint(
        name=serving_endpoint_name,
        config=endpoint_config
    )

# COMMAND ----------

# Wait for Endpoint to be ready

import time
from datetime import datetime, timedelta

# Define the maximum wait time (20 minutes)
max_wait_time = timedelta(minutes=20)
deadline = datetime.now() + max_wait_time

# Function to check the status of the endpoint
def check_endpoint_status(client, endpoint_name):
    endpoints = client.list_endpoints()
    for endpoint in endpoints:
        if endpoint['name'] == endpoint_name:
            return endpoint
    return None

# Wait for the endpoint to be ready or until the deadline is reached
while datetime.now() < deadline:
    endpoint_info = check_endpoint_status(client, serving_endpoint_name)
    if endpoint_info is not None and str(endpoint_info['state']['ready']).lower() == 'ready' and str(endpoint_info['state']['config_update']).lower() != 'in_progress':
        print(f"Endpoint {serving_endpoint_name} is ready.")
        break
    else:
        print(f"Waiting for endpoint {serving_endpoint_name} to be ready. Current status: {endpoint_info['state'] if endpoint_info else 'Not Found'}")
        time.sleep(60)  # Wait for 60 seconds before checking again
else:
    print(f"Timeout reached. Endpoint {serving_endpoint_name} may not be ready.")

displayHTML(
    f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build Forecast and Continuous Improvement
# MAGIC After developing and deploying your machine learning model, the final step is to utilize the model to make predictions. This process involves sending new data to the model endpoint and interpreting the predictions returned by the model.
# MAGIC
# MAGIC - Generate Forecasts: Use your model to predict future values based on historical data.
# MAGIC - Validate and Iterate: Continuously validate your model against new data and iterate to improve accuracy and reliability.

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO: adjust the code to deploy the best model you trained**

# COMMAND ----------

#Predict using Served Model
import requests
import json
from datetime import date, datetime

# Custom encoder for handling date and datetime objects in JSON serialization
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# Prepare data payload from DataFrame for model invocation
data_payload = {"dataframe_records": history_pd.to_dict(orient='records')}
data_json = json.dumps(data_payload, cls=CustomJSONEncoder)



# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# Setup headers for the POST request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}

# Endpoint URL for model invocation
serving_endpoint_url = f"{API_ROOT}/serving-endpoints/{serving_endpoint_name}/invocations"

# API call to deploy model and obtain predictions
response = requests.post(serving_endpoint_url, headers=headers, data=data_json)

# Check and display the response
if response.status_code == 200:
    predictions = response.json()
    print("Predictions:", predictions)
else:
    print("Failed to make predictions")

# COMMAND ----------

#Visualize the Predictions
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Convert predictions JSON to DataFrame
pred_df = pd.json_normalize(predictions['predictions'])

# Ensure 'ds' columns are datetime objects for merging and filtering
history_pd['ds'] = pd.to_datetime(history_pd['ds'])
pred_df['ds'] = pd.to_datetime(pred_df['ds'])

# Merge historical and prediction data on 'ds'
combined_df = pd.merge(left=pred_df, right=history_pd, on='ds', how='left')

# Filter for data from the last 60 days
combined_df = combined_df[combined_df['ds'] >= history_pd['ds'].max() - timedelta(days=60)]

# Plotting setup
plt.figure(figsize=(12, 6))

# Plot actual values and predictions
plt.plot(combined_df['ds'], combined_df['y'], label='Actual', color='black')
plt.plot(combined_df['ds'], combined_df['yhat'], label='Predicted', color='blue')

# Indicate prediction uncertainty
plt.fill_between(combined_df['ds'], combined_df['yhat_lower'], combined_df['yhat_upper'], color='gray', alpha=0.2)

# Finalize plot
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Now its your turn!
# MAGIC
# MAGIC - Iterate model performance
# MAGIC - Build a dashboard showing performance metrics and data insights
# MAGIC - Embed your dashboard into an app that allows the user generate predictions
# MAGIC - ...

# COMMAND ----------


