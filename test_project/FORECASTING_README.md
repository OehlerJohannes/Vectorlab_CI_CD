# Prophet Forecasting Model - Training and Validation Workflow

This directory contains the Databricks Asset Bundle configuration for training and validating a Prophet forecasting model using MLflow and Unity Catalog.

## Project Structure

```
test_project/
├── src/
│   ├── train_forecast_model.ipynb      # Training notebook
│   └── validate_forecast_model.ipynb   # Validation notebook
├── resources/
│   ├── forecast-ml-artifacts-resource.yml        # UC model & experiment setup
│   └── forecast-model-workflow-resource.yml      # Training & validation workflow
├── databricks.yml                       # Bundle configuration
└── FORECASTING_README.md               # This file
```

## Notebooks

### 1. `train_forecast_model.ipynb`
Trains a Prophet forecasting model and registers it to Unity Catalog.

**Key Features:**
- Loads data from Unity Catalog tables
- Trains Prophet model with configurable parameters
- Logs model parameters and metrics to MLflow
- Registers model to Unity Catalog automatically
- Sets task values for downstream workflow tasks

**Parameters:**
- `env`: Environment (dev, staging, prod)
- `catalog`: Catalog name for training data
- `schema`: Schema name for training data
- `table`: Table name for training data
- `forecast_horizon`: Number of periods to forecast
- `experiment_name`: MLflow experiment name
- `model_name`: Three-level UC model name (catalog.schema.model)

### 2. `validate_forecast_model.ipynb`
Validates trained models using `mlflow.evaluate()` before deployment.

**Key Features:**
- Gets model info from training task via `taskValues`
- Loads validation data and prepares it
- Runs `mlflow.evaluate()` with configurable thresholds
- Compares against baseline (champion) model optionally
- Sets "challenger" alias if validation passes
- Logs validation results to model description

**Parameters:**
- `run_mode`: disabled/dry_run/enabled
- `enable_baseline_comparison`: Compare against champion model
- `catalog/schema/table`: Validation data location
- `forecast_horizon`: Forecast horizon
- `model_name`: UC model name
- `model_version`: Model version to validate
- `experiment_name`: MLflow experiment

## Resource Files

### `forecast-ml-artifacts-resource.yml`
Defines the Unity Catalog model and MLflow experiment infrastructure.

**Creates:**
- Registered model: `{catalog}.vectorlab.prophet_forecast`
- MLflow experiment: `/{env}-prophet-forecast-experiment`
- Permissions for users to read experiments and execute models

### `forecast-model-workflow-resource.yml`
Defines the Databricks job workflow with two tasks:

**Task 1: Train**
- Runs training notebook
- Registers model to Unity Catalog
- Passes model info to validation task

**Task 2: ModelValidation** (depends on Train)
- Validates the trained model
- Sets "challenger" alias if validation passes
- Blocks deployment if validation fails (in enabled mode)

**Schedule:**
- Weekly on Mondays at 6am UTC
- Starts paused (can be enabled after testing)

## Deployment Steps

### 1. Prerequisites
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
databricks configure --token
```

### 2. Validate Configuration
```bash
cd /path/to/test_project
databricks bundle validate
```

### 3. Deploy to Dev Environment
```bash
# Deploy infrastructure and workflows
databricks bundle deploy -t dev

# This creates:
# - Unity Catalog model: johannes_oehler.vectorlab.prophet_forecast
# - MLflow experiment: /dev-prophet-forecast-experiment
# - Databricks job: dev-prophet-forecast-training-job
```

### 4. Run the Workflow

**Option A: Trigger via CLI**
```bash
databricks bundle run prophet_forecast_training_job -t dev
```

**Option B: Trigger via Databricks UI**
1. Go to Workflows in Databricks workspace
2. Find `dev-prophet-forecast-training-job`
3. Click "Run now"

**Option C: Enable Schedule**
- Edit `forecast-model-workflow-resource.yml`
- Change `pause_status: PAUSED` to `pause_status: UNPAUSED`
- Redeploy: `databricks bundle deploy -t dev`

### 5. Monitor Execution

**Check Job Status:**
```bash
databricks jobs list | grep prophet-forecast
databricks jobs runs list --job-id <job-id>
```

**View in UI:**
- Workflows → `dev-prophet-forecast-training-job` → View runs
- MLflow → `/dev-prophet-forecast-experiment` → View training runs
- Unity Catalog → `johannes_oehler.vectorlab.prophet_forecast` → View model versions

## Model Lifecycle

### 1. Training
- Model is trained and logged to MLflow
- Automatically registered to Unity Catalog
- Version number auto-incremented

### 2. Validation
- Model is validated using `mlflow.evaluate()`
- Metrics compared against thresholds
- If passed: "challenger" alias is set
- If failed: Deployment blocked (in enabled mode)

### 3. Deployment (Manual Step)
After validation passes, promote to champion:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
model_name = "johannes_oehler.vectorlab.prophet_forecast"
version = "2"  # The validated version

# Set champion alias
client.set_registered_model_alias(model_name, "champion", version)
```

### 4. Serving (Optional)
Use the champion model for inference:
```python
import mlflow

mlflow.set_registry_uri('databricks-uc')
model = mlflow.pyfunc.load_model("models:/johannes_oehler.vectorlab.prophet_forecast@champion")

# Make predictions
predictions = model.predict(data)
```

## Configuration

### Validation Thresholds
Edit in `validate_forecast_model.ipynb` cell 7:
```python
validation_thresholds = {
    "mean_squared_error": MetricThreshold(threshold=1000, greater_is_better=False),
    "mean_absolute_error": MetricThreshold(threshold=25, greater_is_better=False),
    "root_mean_squared_error": MetricThreshold(threshold=30, greater_is_better=False),
}
```

### Run Mode
Set in workflow base_parameters:
- `disabled`: Skip validation entirely
- `dry_run`: Run validation but don't block on failure (default)
- `enabled`: Block deployment if validation fails

### Schedule
Edit in `forecast-model-workflow-resource.yml`:
```yaml
schedule:
  quartz_cron_expression: "0 0 6 ? * MON"  # Weekly on Mondays
  timezone_id: UTC
  pause_status: PAUSED
```

## Troubleshooting

### Job Fails with "Model not found"
- Ensure `forecast-ml-artifacts-resource.yml` is deployed
- Check Unity Catalog model exists: `johannes_oehler.vectorlab.prophet_forecast`

### Validation Task Can't Find Model Version
- Check Train task completed successfully
- Verify model was registered in Train task output
- Check task values are being passed correctly

### Permission Errors
- Ensure you have permissions on the catalog/schema
- Check grants in `forecast-ml-artifacts-resource.yml`
- Verify your user has appropriate permissions

### Data Not Found
- Update table location in workflow parameters
- Ensure table exists: `johannes_oehler.vectorlab.forecast_data`
- Check query in notebooks

## Best Practices

1. **Test Notebooks Standalone First**
   - Run notebooks manually in Databricks
   - Verify data loading and model training
   - Then integrate into workflow

2. **Use dry_run Mode Initially**
   - Test validation logic without blocking
   - Adjust thresholds based on results
   - Switch to enabled mode when ready

3. **Version Control**
   - Commit all changes to Git
   - Bundle deployment tracks Git info automatically
   - Model versions linked to code versions

4. **Monitor Metrics**
   - Check MLflow experiments regularly
   - Set up alerts for validation failures
   - Track model performance over time

5. **Incremental Updates**
   - Deploy to dev first
   - Test thoroughly
   - Deploy to prod when stable

## Next Steps

1. ✅ Deploy bundle: `databricks bundle deploy -t dev`
2. ✅ Run workflow manually to test
3. ✅ Adjust validation thresholds based on results
4. ✅ Enable schedule when ready for production
5. ✅ Add deployment task (optional)
6. ✅ Set up monitoring and alerts

## Resources

- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [MLflow Evaluate](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate)
- [Prophet Documentation](https://facebook.github.io/prophet/)

