from .config import ModelConfig
from zenml.client import Client
from zenml import step
import mlflow, logging
from tqdm import tqdm
from torch import nn, optim
import torch, sys
from torch.utils.data import DataLoader
from module.model_dev import StyleTransferVAE
from module.evaluation import VAE_Loss, ResNetDiscriminator, Adversarial_Loss, EdgeAccuracyLoss, PerceptualLoss, ContentLoss, StyleLoss
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import numpy as np

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_genres: int,
    config: ModelConfig,
) -> nn.Module:
    try:
        if config.model_name == "StyleTransferVAE":
            
            mlflow.pytorch.autolog()
            
            best_val_loss = np.inf
            patience = 3
            patience_counter = 0
            
            total_train_loss = 0.0
            total_rec_train_loss = 0.0
            total_kl_train_loss = 0.0
            # total_disc_train_loss = 0.0
            # total_perp_train_loss = 0.0
            # total_edge_train_loss = 0.0
            # total_content_train_loss = 0.0
            # total_style_train_loss = 0.0
            
            num_batches = 0
            
            total_val_loss = 0.0
            total_rec_val_loss = 0.0
            total_kl_val_loss = 0.0
            # total_disc_val_loss = 0.0
            # total_perp_val_loss = 0.0
            # total_edge_val_loss = 0.0
            # total_content_val_loss = 0.0
            # total_style_val_loss = 0.0
            
            style_vae_model = StyleTransferVAE(
                z_dim=config.Z_DIM,
                num_genres=n_genres,
                img_size=config.IMG_SIZE
            ).to(config.DEVICE)
            
            # resnet_disc = ResNetDiscriminator(
            #     z_dim=config.Z_DIM,
            #     num_genres=n_genres
            # ).to(config.DEVICE)
            
            with open("style_vae_model_summary.txt", "w", encoding="utf-8") as f:
                f.write(str(summary(style_vae_model)))
            mlflow.log_artifact("style_vae_model_summary.txt")
            
            # with open("resnet_disc_model_summary.txt", "w", encoding="utf-8") as f:
            #     f.write(str(summary(resnet_disc)))
            # mlflow.log_artifact("resnet_disc_model_summary.txt")

            vae_optimizer = optim.Adam(style_vae_model.parameters(), lr=config.STYLE_VAE_LR_RATE)
            # disc_optimizer = optim.Adam(resnet_disc.parameters(), lr=config.RESNET_DISC_LR_RATE)
            
            vae_scheduler = StepLR(vae_optimizer, step_size=5, gamma=0.1)
            # disc_scheduler = StepLR(disc_optimizer, step_size=30, gamma=0.1)
            
            loss_fn = VAE_Loss()
            # adv_loss_fn = Adversarial_Loss()
            # edge_acc_loss_fn = EdgeAccuracyLoss().to(config.DEVICE)
            # perp_loss_fn = PerceptualLoss().to(config.DEVICE)
            # style_loss_fn = StyleLoss().to(config.DEVICE)
            # content_loss_fn = ContentLoss().to(config.DEVICE)
            
            params = {
                "epochs": config.NUM_EPOCHS,
                "style_vae_learning_rate": config.STYLE_VAE_LR_RATE,
                "resnet_disc_learning_rate": config.RESNET_DISC_LR_RATE,
                "batch_size": config.BATCH_SIZE,
                "loss_function": loss_fn.__class__.__name__,
                "style_vae_optimizer": vae_optimizer.__class__.__name__,
                # "disc_optimizer": disc_optimizer.__class__.__name__,
            }
            # Log training parameters.
            mlflow.log_params(params)
            
            if config.LOAD_MODEL:
                style_vae_model.load_state_dict(torch.load('temp_style_vae_model.pth', map_location=config.DEVICE))
                # resnet_disc.load_state_dict(torch.load('temp_resnet_disc_model.pth', map_location=config.DEVICE))
                vae_optimizer.load_state_dict(torch.load('temp_style_vae_optimizer.pth', map_location=config.DEVICE))
                # disc_optimizer.load_state_dict(torch.load('temp_resnet_disc_optimizer.pth', map_location=config.DEVICE))
            
            for epoch in range(config.NUM_EPOCHS):
                style_vae_model.train()
                # resnet_disc.train()
                
                train_loop = tqdm(train_dataloader)
                for images, genres in train_loop:
                    images, genres = images.to(config.DEVICE), genres.to(config.DEVICE)
                    reconstructed_x, mu, sigma, style_z = style_vae_model(images, genres)
                    
                    rec_train_loss, kl_train_loss = loss_fn(reconstructed_x, images, mu, sigma, ret_components=True)
                    kl_train_loss = config.BVAE * (kl_train_loss / config.BATCH_SIZE)
                    # adv_train_loss = adv_loss_fn(resnet_disc, style_z.detach(), genres)
                    # edge_train_loss = edge_acc_loss_fn(reconstructed_x, images)
                    # perp_train_loss = perp_loss_fn(reconstructed_x, images)
                    # style_train_loss = style_loss_fn(reconstructed_x, images)
                    # content_train_loss = content_loss_fn(reconstructed_x, images)
                    
                    train_loss = config.LAMBDA_REC_LOSS * rec_train_loss + \
                                 config.LAMBDA_KL_LOSS * kl_train_loss 
                                #  config.LAMBDA_DISC_LOSS * adv_train_loss + \
                                #  edge_train_loss + perp_train_loss + style_train_loss + content_train_loss
                                        
                    vae_optimizer.zero_grad()
                    train_loss.backward()
                    vae_optimizer.step()
                    
                    # disc_optimizer.zero_grad()
                    # genre_predictions = resnet_disc(style_z.detach())
                    # disc_train_loss = F.binary_cross_entropy_with_logits(genre_predictions, genres)
                    # disc_train_loss.backward()
                    # disc_optimizer.step()
                    
                    total_train_loss += train_loss.item()
                    total_rec_train_loss +=(config.LAMBDA_REC_LOSS * rec_train_loss.item())
                    total_kl_train_loss +=(config.LAMBDA_KL_LOSS *  kl_train_loss.item())
                    # total_edge_train_loss +=(edge_train_loss.item())
                    # total_perp_train_loss +=(perp_train_loss.item())
                    # total_style_train_loss +=(style_train_loss.item())
                    # total_content_train_loss +=(content_train_loss.item())
                    # total_disc_train_loss +=(config.LAMBDA_DISC_LOSS * disc_train_loss.item())
                    num_batches += 1
                    
                    train_loop.set_postfix(
                        train_l=f"{train_loss.item():.4f}", 
                        rec_l=f"{config.LAMBDA_REC_LOSS * rec_train_loss.item():.4f}", 
                        kl_l=f"{config.LAMBDA_KL_LOSS * kl_train_loss.item():.4f}", 
                        # edge_l=f"{edge_train_loss.item():.4f}", 
                        # perp_l=f"{perp_train_loss.item():.4f}",
                        # style_l=f"{style_train_loss.item():.4f}",
                        # content_l=f"{content_train_loss.item():.4f}",
                        # disc_l=f"{config.LAMBDA_DISC_LOSS * disc_train_loss.item():.4f}", 
                        epoch=epoch, 
                        run_type='train'
                    )
                
                avg_train_loss = total_train_loss / num_batches
                avg_rec_train_loss = total_rec_train_loss / num_batches
                avg_kld_train_loss = total_kl_train_loss / num_batches
                # avg_edge_train_loss = total_edge_train_loss / num_batches
                # avg_perp_train_loss = total_perp_train_loss / num_batches
                # avg_style_train_loss = total_style_train_loss / num_batches
                # avg_content_train_loss = total_content_train_loss / num_batches
                # avg_disc_train_loss = total_disc_train_loss / num_batches
                
                total_train_loss = 0.0
                total_rec_train_loss = 0.0
                total_kl_train_loss = 0.0
                # total_disc_train_loss = 0.0
                # total_edge_train_loss = 0.0
                # total_perp_train_loss = 0.0
                # total_style_train_loss = 0.0
                # total_content_train_loss = 0.0
                num_batches = 0
                
                mlflow.log_metric("style_vae_train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("rec_train_loss", avg_rec_train_loss, step=epoch)
                mlflow.log_metric("kl_div_train_loss", avg_kld_train_loss, step=epoch)
                # mlflow.log_metric("edge_train_loss", avg_edge_train_loss, step=epoch)
                # mlflow.log_metric("perp_train_loss", avg_perp_train_loss, step=epoch)
                # mlflow.log_metric("style_train_loss", avg_style_train_loss, step=epoch)
                # mlflow.log_metric("content_train_loss", avg_content_train_loss, step=epoch)
                # mlflow.log_metric("disc_train_loss", avg_disc_train_loss, step=epoch)
                
                style_vae_model.eval()
                # resnet_disc.eval()
                
                val_loop = tqdm(val_dataloader)
                with torch.no_grad():
                    for images, genres in val_loop:
                        images, genres = images.to(config.DEVICE), genres.to(config.DEVICE)
                        reconstructed_x, mu, sigma, style_z = style_vae_model(images, genres)
                        
                        rec_val_loss, kl_val_loss = loss_fn(reconstructed_x, images, mu, sigma, ret_components=True)
                        kl_val_loss = config.BVAE * (kl_val_loss / config.BATCH_SIZE)
                        # adv_val_loss = adv_loss_fn(resnet_disc, style_z.detach(), genres)
                        # edge_val_loss = edge_acc_loss_fn(reconstructed_x, images)
                        # perp_val_loss = perp_loss_fn(reconstructed_x, images)
                        # style_val_loss = style_loss_fn(reconstructed_x, images)
                        # content_val_loss = content_loss_fn(reconstructed_x, images)
                        
                        val_loss = config.LAMBDA_REC_LOSS * rec_val_loss + \
                                    config.LAMBDA_KL_LOSS * kl_val_loss 
                                    # config.LAMBDA_DISC_LOSS * adv_val_loss + \
                                    # edge_val_loss + perp_val_loss + style_val_loss + content_val_loss
                        
                        # genre_predictions = resnet_disc(style_z.detach())
                        # disc_val_loss = F.binary_cross_entropy_with_logits(genre_predictions, genres)
                        
                        total_val_loss += val_loss.item()
                        total_rec_val_loss +=(config.LAMBDA_REC_LOSS * rec_val_loss.item())
                        total_kl_val_loss +=(config.LAMBDA_KL_LOSS *  kl_val_loss.item())
                        # total_edge_val_loss +=(edge_val_loss.item())
                        # total_perp_val_loss +=(perp_val_loss.item())
                        # total_style_val_loss +=(style_val_loss.item())
                        # total_content_val_loss +=(content_val_loss.item())
                        # total_disc_val_loss +=(config.LAMBDA_DISC_LOSS * disc_val_loss.item())
                        num_batches += 1
                    
                        val_loop.set_postfix(
                            val_l=f"{val_loss.item():.4f}", 
                            rec_l=f"{config.LAMBDA_REC_LOSS * rec_val_loss.item():.4f}", 
                            kl_l=f"{config.LAMBDA_KL_LOSS * kl_val_loss.item():.4f}", 
                            # edge_l=f"{edge_val_loss.item():.4f}", 
                            # perp_l=f"{perp_val_loss.item():.4f}", 
                            # style_l=f"{style_val_loss.item():.4f}",
                            # content_l=f"{content_val_loss.item():.4f}",
                            # disc_l=f"{config.LAMBDA_DISC_LOSS * disc_val_loss.item():.4f}", 
                            epoch=epoch, 
                            run_type='val'
                        )
                
                avg_val_loss = total_val_loss / num_batches
                avg_rec_val_loss = total_rec_val_loss / num_batches
                avg_kld_val_loss = total_kl_val_loss / num_batches
                # avg_edge_val_loss = total_edge_val_loss / num_batches
                # avg_perp_val_loss = total_perp_val_loss / num_batches
                # avg_style_val_loss = total_style_val_loss / num_batches
                # avg_content_val_loss = total_content_val_loss / num_batches
                # avg_disc_val_loss = total_disc_val_loss / num_batches
                
                total_val_loss = 0.0
                total_rec_val_loss = 0.0
                total_kl_val_loss = 0.0
                # total_edge_val_loss = 0.0
                # total_perp_val_loss = 0.0
                # total_disc_val_loss = 0.0
                # total_style_val_loss = 0.0
                # total_content_val_loss = 0.0
                num_batches = 0
                
                mlflow.log_metric("style_vae_val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("rec_val_loss", avg_rec_val_loss, step=epoch)
                mlflow.log_metric("kl_div_val_loss", avg_kld_val_loss, step=epoch)
                # mlflow.log_metric("edge_val_loss", avg_edge_val_loss, step=epoch)
                # mlflow.log_metric("perp_val_loss", avg_perp_val_loss, step=epoch)
                # mlflow.log_metric("style_val_loss", avg_style_val_loss, step=epoch)
                # mlflow.log_metric("content_val_loss", avg_content_val_loss, step=epoch)
                # mlflow.log_metric("disc_val_loss", avg_disc_val_loss, step=epoch)
                
                torch.save(style_vae_model.state_dict(), 'temp_style_vae_model.pth')
                # torch.save(resnet_disc.state_dict(), 'temp_resnet_disc_model.pth')
                torch.save(vae_optimizer.state_dict(), 'temp_style_vae_optimizer.pth')
                # torch.save(disc_optimizer.state_dict(), 'temp_resnet_disc_optimizer.pth')
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save model checkpoint if desire
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Stopping early at epoch {epoch}")
                        break 
                
                vae_scheduler.step()
                # disc_scheduler.step()
                
            trained_model = style_vae_model
            
            mlflow.pytorch.log_model(style_vae_model, "style_vae_model")
            # mlflow.pytorch.log_model(resnet_disc, "resnet_disc")
            
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        _, _, line = sys.exc_info()
        logging.error(f"Error in training model: {e}. Line No.{line.tb_lineno}")
        raise e