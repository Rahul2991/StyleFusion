from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.model_train import train_model
from steps.eval import evaluate_model

@pipeline
def train_pipeline(data_root: str, csv_file: str):
    dataset, n_genres = ingest_data(csv_file, data_root)
    train_dataloader, val_dataloader, test_dataloader = preprocess_data(dataset)
    model = train_model(train_dataloader, val_dataloader, n_genres)
    avg_loss, avg_rec_loss, avg_kld_loss, avg_fid_score = evaluate_model(model, test_dataloader)