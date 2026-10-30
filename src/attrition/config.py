import os

DATA_URL = os.getenv(
    "DATA_URL",
    "https://raw.githubusercontent.com/ryankarlos/Human-Resource-Analytics-Kaggle-Dataset/master/Original_Kaggle_Dataset/HR_comma_sep.csv",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "employee-attrition")
MODEL_NAME = os.getenv("MODEL_NAME", "employee-attrition")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

TARGET = "attrition"
NUMERICAL = [
    "satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
    "time_spend_company", "Work_accident", "promotion_last_5years",
]
CATEGORICAL = ["dept", "salary"]
FEATURES = NUMERICAL + CATEGORICAL
TEST_SIZE = 0.1
RANDOM_STATE = 0
