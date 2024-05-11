from zenml.steps import BaseParameters
# from pydantic import BaseModel
import torch

class ModelConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "StyleTransferVAE"
    DEVICE: str = "cuda" if torch.cuda.is_available() else 'cpu'
    IMG_SIZE: int = 256
    IMG_CHANNELS: int = 3
    INPUT_DIM: int = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
    H_DIM: int  = 200
    Z_DIM: int = 256
    NUM_EPOCHS: int  = 100
    BATCH_SIZE: int  = 32
    STYLE_VAE_LR_RATE: float  = 1e-4
    RESNET_DISC_LR_RATE: float  = 3e-10
    FID_CALC_FREQ: int  = 1
    LAMBDA_REC_LOSS: float = 1
    LAMBDA_KL_LOSS: float = 1
    LAMBDA_PERP_LOSS: float = 1.0
    LAMBDA_EDGE_LOSS: float = 1.0
    # LAMBDA_DISC_LOSS: float = 0.1
    BVAE: float = 1

    LOAD_MODEL: bool = True