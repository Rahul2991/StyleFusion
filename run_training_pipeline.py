from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == '__main__':
    print(f'Tracking URI (MLFLOW): {get_tracking_uri()}')
    train_pipeline(data_root='data/images/images', csv_file='data/artists.csv')