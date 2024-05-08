import logging, torch, sys
from torch import nn, Tensor
from pytorch_fid.fid_score import calculate_fid_given_paths
from steps.config import ModelConfig
from torchvision import models
from torch.nn import functional as F

class Reconstruction_Loss(nn.Module):
    """
    Evaluation strategy that uses BCE Loss to calculate Reconstruction Loss
    """   
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, reconstructed_x: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            reconstructed_x: Tensor
            x: Tensor
        Returns:
            reconstruction_loss: Tensor
        """
        try: 
            return F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        except Exception as e:
            logging.error(
                "Exception occurred in calculate reconstruction loss. Exception message:  "
                + str(e)
            )
            raise e
        
class KL_Divergence_Loss(nn.Module):
    """
    Evaluation strategy that calculate KL Divergence Loss
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Args:
            mu: Tensor
            sigma: Tensor
        Returns:
            kl_divergence_loss: Tensor
        """
        try:
            # logging.info("Entered the calculate kl Divergence loss")
            # Maximize in paper but minus in front is to minimize loss with pytorch ## For standard gaussian
            # kl_divergence_loss = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) 
            
            # logging.info("The KL Divergence Loss value is: " + str(kl_divergence_loss))
            return 0.5 * torch.sum(mu.pow(2) + torch.exp(sigma) - sigma - 1)
        except Exception as e:
            logging.error(
                "Exception occurred in calculate kl Divergence loss. Exception message:  "
                + str(e)
            )
            raise e
        
class Adversarial_Loss(nn.Module):
    """
    Evaluation strategy that calculate Adversarial Loss
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, discriminator: nn.Module, style_z: Tensor, true_labels: Tensor) -> Tensor:
        """
        Args:
            discriminator: Tensor
            style_z: Tensor
            true_labels: Tensor
        Returns:
            adv_loss: Tensor
        """
        try:
            predictions = discriminator(style_z)
            return F.binary_cross_entropy_with_logits(predictions, true_labels)
        except Exception as e:
            logging.error(
                "Exception occurred in calculate Adversarial loss. Exception message:  "
                + str(e)
            )
            raise e
        
class VAE_Loss(nn.Module):
    """
    Evaluation strategy that uses Reconstruction_Loss (BCE) + KL Divergence to calculate ELBO / VAE Loss
    """
    def __init__(self) -> None:
        super().__init__() 
        self.rec_loss = Reconstruction_Loss()
        self.kl_loss = KL_Divergence_Loss()
               
    def forward(self, reconstructed_x: Tensor, x: Tensor, mu: Tensor, sigma: Tensor, ret_components: bool = False) -> Tensor:
        """
        Args:
            reconstructed_x: Tensor
            x: Tensor
            mu: Tensor
            sigma: Tensor
            ret_components: bool (Optional)
        Returns:
            elbo_loss: Tensor
        """
        try:
            rec_loss = self.rec_loss(reconstructed_x, x)
            kl_loss = self.kl_loss(mu, sigma)
            if ret_components:
                return rec_loss, kl_loss
            else:
                elbo_loss = rec_loss + kl_loss
                return elbo_loss
        except Exception as e:
            logging.error(
                "Exception occurred in ELBO Loss method of the VAE_Loss class. Exception message:  "
                + str(e)
            )
            raise e
        
class FID_Score(nn.Module):
    """
    Evaluation strategy that uses FID_Score to calculate similarity between reconstructed and actual image
    """
    def __init__(self) -> None:
        super().__init__() 
        
        self.config = ModelConfig()
        
    def forward(self, real_images_path: str, generated_images_path: str):
        """
        Args:
            real_images_path: str
            generated_images_path: str
        """
        try:
            # logging.info("Entered the calculation of FID Score method")
            fid_score = calculate_fid_given_paths([real_images_path, generated_images_path],
                                          batch_size=self.config.BATCH_SIZE,
                                          device=self.config.DEVICE,
                                          dims=2048, # inception v3 vector dim
                                          num_workers=0)
            # logging.info("The FID Score value is: " + str(fid_score))
            return fid_score
        except Exception as e:
            logging.error(
                "Exception occurred in the calculation of FID Score method. Exception message:  "
                + str(e)
            )
            raise e
        
class ResNetDiscriminator(nn.Module):
    def __init__(self, z_dim, num_genres):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replaceing the first convolution layer to accept 1D input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Adjusting the fully connected layer to match the number of genres
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_genres)

        # Additional layer to transform style_z into a format suitable for Conv2d input
        self.input_transform = nn.Linear(z_dim, 256)

    def forward(self, style_z):
        try:
            # Transform style_z to a suitable shape
            x = self.input_transform(style_z)
            x = x.view(-1, 1, 16, 16)  # Reshape to (batch_size, channels, height, width)
            
            # Passing through the modified ResNet-18
            return self.resnet(x)
        except Exception as e:
            logging.error(
                "Exception occurred in the ResNetDiscriminator. Exception message:  "
                + str(e)
            )
            raise e

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:23].eval()  # Using up to the third block
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        try:
            vgg_input = self.vgg(input)
            vgg_target = self.vgg(target)
            loss = self.criterion(vgg_input, vgg_target)
            return loss
        except Exception as e:
            _, _, line = sys.exc_info()
            logging.error(
                f"Exception occurred in the PerceptualLoss. Exception message:  {str(e)}. Line No.{line.tb_lineno}"
            )
            raise e
        
class EdgeDetection(nn.Module):
    def __init__(self):
        super(EdgeDetection, self).__init__()
        # Sobel filter kernels for 3-channel input
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)  # Apply filter to each channel independently
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)

        # Initialize the Sobel filter weights for edge detection in the x and y directions
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3,1,1,1)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3,1,1,1)

        # Assign the kernels to the convolutional layers
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel, requires_grad=False)

    def forward(self, x):
        try:
            x = x.unsqueeze(1) if x.dim() == 3 else x
            edge_x = self.sobel_x(x)
            edge_y = self.sobel_y(x)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
            return edge
        except Exception as e:
            _, _, line = sys.exc_info()
            logging.error(
                f"Exception occurred in the EdgeDetection. Exception message:  {str(e)}. Line No.{line.tb_lineno}"
            )
            raise e
    
class EdgeAccuracyLoss(nn.Module):
    def __init__(self):
        super(EdgeAccuracyLoss, self).__init__()
        self.edge_detector = EdgeDetection()
        self.loss_fn = nn.MSELoss()

    def forward(self, generated, target):
        try:
            # Generating edge maps
            edge_gen = self.edge_detector(generated)
            edge_tar = self.edge_detector(target)

            # Calculating loss using the MSE loss module
            loss = self.loss_fn(edge_gen, edge_tar)
            return loss
        except Exception as e:
            _, _, line = sys.exc_info()
            logging.error(
                f"Exception occurred in the EdgeAccuracyLoss. Exception message:  {str(e)}. Line No.{line.tb_lineno}"
            )
            raise e
    
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights='DEFAULT').features
        
        # Freezing all VGG parameters since we're only using it for feature extraction
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Using the relu4_2 layer to extract features
        self.vgg_layers = vgg[:21]  # Up to and including relu4_2
        self.vgg_layers.eval()  # Setting to eval mode to deactivate dropout and other training-specific layers
        
        self.loss_fn = nn.MSELoss()

    def forward(self, generated, target):
        generated_features = self.vgg_layers(generated)
        target_features = self.vgg_layers(target)

        return self.loss_fn(generated_features, target_features)

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)
    
class StyleLoss(nn.Module):
    def __init__(self, feature_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super(StyleLoss, self).__init__()
        self.vgg = models.vgg19(weights='DEFAULT').features
        # Freezing all VGG parameters since we're only using it for feature extraction
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Only need to forward until the highest layer in feature_layers
        self.max_layer = max([int(layer.split('_')[1]) for layer in feature_layers])
        self.feature_layers = feature_layers
        self.gram = GramMatrix()
        self.loss_fn = nn.MSELoss()

    def forward(self, generated, target):
        gen_features = []
        tar_features = []
        x = generated
        y = target
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if f'relu{i//2 + 1}_{i%2 + 1}' in self.feature_layers:
                gen_features.append(self.gram(x))
                tar_features.append(self.gram(y))
            if i // 2 + 1 > self.max_layer:
                break

        style_loss = 0
        for gen_g, tar_g in zip(gen_features, tar_features):
            style_loss += self.loss_fn(gen_g, tar_g)

        return style_loss
    
if __name__ == '__main__':
    z_dim = 20
    num_genres = 10
    style_z = torch.randint(0, 1, (2, 20)).float()  # (batch, z_dim)
    discriminator = ResNetDiscriminator(z_dim, num_genres)
    print(discriminator(style_z).shape)
    print(discriminator(style_z))