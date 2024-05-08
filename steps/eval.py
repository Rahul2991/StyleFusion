from zenml.client import Client
from torch.utils.data import DataLoader
from torch import nn
from zenml import step
import logging, mlflow, torch
from steps.config import ModelConfig
from tqdm import tqdm
from typing_extensions import Annotated
from typing import Tuple
from utils import save_images
from module.evaluation import VAE_Loss, Reconstruction_Loss, KL_Divergence_Loss, FID_Score

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=True)
def evaluate_model(
    model: nn.Module, 
    test_dataloader: DataLoader, 
    config: ModelConfig
    )-> Tuple[
        Annotated[float, "avg_loss"], 
        Annotated[float, "avg_rec_loss"], 
        Annotated[float, "avg_kld_loss"], 
        Annotated[float, "avg_fid_score"]
    ]:
    """
    Args:
        model: nn.Module
        test_dataloader: DataLoader
        config: ModelConfig
    Returns:
        avg_loss: float
        avg_rec_loss: float
        avg_kld_loss: float
        avg_fid_score: float
        avg_llh: float
    """
    try:
        if config.model_name == "StyleTransferVAE":
            model.eval()
            total_loss = 0.0
            total_rec_loss = 0.0
            total_kld_loss = 0.0
            total_fid_score = 0.0
            num_batches = 0
            loss_fn = VAE_Loss()
            rec_loss_fn = Reconstruction_Loss()
            kl_div_loss_fn = KL_Divergence_Loss()
            fid_score_fn = FID_Score()
            fid_score=None
            
            test_loop = tqdm(test_dataloader)
            with torch.no_grad():
                for images, genres in test_loop:
                    images, genres = images.to(config.DEVICE), genres.to(config.DEVICE)
                    reconstructed_x, mu, sigma, style_z = model(images, genres)
                    
                    test_loss = loss_fn(reconstructed_x, images, mu, sigma)
                    rec_loss = rec_loss_fn(reconstructed_x, images)
                    kl_div_loss = kl_div_loss_fn(mu, sigma)
                    
                    total_loss += test_loss.item()
                    total_rec_loss +=rec_loss.item()
                    total_kld_loss +=kl_div_loss.item()
                    num_batches += 1
                    
                    test_loop.set_postfix(
                        vae_loss=f"{test_loss.item():.4f}", 
                        rec_loss=f"{rec_loss.item():.4f}", 
                        kl_div_loss=f"{kl_div_loss.item():.4f}", 
                        fid_score=0 if fid_score is None else fid_score.item(), 
                        run_type='test'
                    )
                    
                save_images(reconstructed_x, 'data/rec_imgs/')
                save_images(images, 'data/original_imgs/')
                fid_score = fid_score_fn('data/original_imgs/', 'data/rec_imgs/')
                total_fid_score +=fid_score.item()
                    
                avg_loss = total_loss / num_batches
                avg_rec_loss = total_rec_loss / num_batches
                avg_kld_loss = total_kld_loss / num_batches
                avg_fid_score = total_fid_score / num_batches
                
                total_loss = 0.0
                total_rec_loss = 0.0
                total_kld_loss = 0.0
                total_fid_score = 0.0
                num_batches = 0
                
                mlflow.log_metric("vae_loss", avg_loss)
                mlflow.log_metric("rec_loss", avg_rec_loss)
                mlflow.log_metric("kl_div_loss", avg_kld_loss)
                mlflow.log_metric("fid_score", avg_fid_score)
                
            return avg_loss, avg_rec_loss, avg_kld_loss, avg_fid_score
        else:
            raise ValueError(f"Model {config.model_name} not supported") 
    except Exception as e:
        logging.error(f"Error in evaluation model: {e}")
        raise e