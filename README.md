# Style Transfer VAE (CVAE) - Pytorch

This project implements a Style Transfer VAE (CVAE) using Pytorch, aimed at transforming images into specific painting genres from the "Best Artworks of all Time" dataset. The pipeline is structured with ZenML, and experiments are tracked using MLFlow.

## Installation

### Prerequisites

- Python 3.10

### Install Pytorch and Related Packages

Use Conda to install Pytorch and related libraries:

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Install Other Dependencies

Install other required Python libraries:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the dataset from Kaggle:
[Best Artworks of all Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

### Dataset Structure

After downloading, extract the dataset into the `data` folder within this repository. Ensure the following structure:

```
data/
    ├── Images/         # Original images
    ├── resized/        # Resized images
    └── artists.csv     # Metadata file
```

## Configuration and Setup

### Initialize ZenML (First Time Only)

```bash
zenml init
```

### Set Up MLFlow for Experiment Tracking 

Install and configure MLFlow with ZenML: (First Time Only)

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

Set the MLFlow stack:

```bash
zenml stack set mlflow_stack
```

Start ZenML:

- For Windows:
  ```bash
  zenml up --blocking
  ```

- For Linux/Mac:
  ```bash
  zenml up
  ```

## Training and Evaluation

Run the training and evaluation pipeline:

```bash
python run_training_pipeline.py
```

After execution, a MLFlow tracking URI will be generated. To access the MLFlow UI:

```bash
mlflow ui --backend-store-uri {YOUR_TRACKING_URI_HERE}
```

## Stop ZenML

To shut down ZenML gracefully, use the following command:

```bash
zenml down
 ```
