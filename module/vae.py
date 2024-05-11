import torch
from torch import nn
from torch.nn import functional as F

class StyleTransferVAE(nn.Module):
    def __init__(self, num_genres=10, z_dim=20, img_size=256):
        super().__init__()
        
        self.img_size = img_size

        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        
        # Fully connected layers for VAE z parameters
        self.fc_mu = nn.Linear(256 * self.img_size // 8 * self.img_size // 8, z_dim)
        self.fc_sigma = nn.Linear(256 * self.img_size // 8 * self.img_size // 8, z_dim)

        # Decoder
        self.dec_fc = nn.Linear(2 * z_dim, 256 * self.img_size // 8 * self.img_size // 8)
        self.dec_conv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.dec_conv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.dec_conv3 = nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid())
        
        # Style Encoder for multiple genres
        self.style_encoder = nn.Sequential(
            nn.Linear(num_genres, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.Sigmoid()
        )
        
    def encode_style(self, genres):
        return self.style_encoder(genres)

    def encode_content(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        mu = self.fc_mu(x3)
        sigma = self.fc_sigma(x3)
        return mu, sigma, x1, x2, x3

    def decode(self, z, x1, x2):
        x = self.dec_fc(z)
        x = x.view(-1, 256, self.img_size // 8, self.img_size // 8)  # Reshape for convolution
        x = self.dec_conv1(x)
        x = self.dec_conv2(x + x2) # Skip connection
        x = self.dec_conv3(x + x1) # Skip connection
        return x

    def forward(self, x, genres):
        mu, sigma, x1, x2, x3 = self.encode_content(x)
        style_z = self.encode_style(genres)
        epsilon = torch.randn_like(sigma)
        content_z = mu + epsilon * torch.exp(sigma / 2)
        combined_z = torch.cat([content_z, style_z], dim=1)
        reconstructed_x = self.decode(combined_z, x1, x2)
        return reconstructed_x, mu, sigma, style_z
    
if __name__ == '__main__':
    img_size = 256
    img_ch = 3
    genres = 10
    x = torch.rand(4, img_ch, img_size, img_size)
    
    print(torch.min(x))
    print(torch.max(x))
    
    g = torch.randint(0, 1, (4, genres)).float()
    vae = StyleTransferVAE(num_genres=genres, z_dim=20)
    reconstructed_x, mu, sigma, style_z = vae(x, g)
    
    print(torch.min(reconstructed_x))
    print(torch.max(reconstructed_x))
    
    print(reconstructed_x.shape)
    print(mu.shape)
    print(sigma.shape)
    print(style_z.shape)
    
    print(type(reconstructed_x))
    print(type(mu))
    print(type(sigma))
    print(type(style_z))