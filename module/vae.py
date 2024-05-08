import torch
from torch import nn

class StyleTransferVAE(nn.Module):
    def __init__(self, num_genres=10, z_dim=20, img_size=256):
        super().__init__()

        # Encoder
        self.content_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * img_size//8 * img_size//8, z_dim)  # Fully connected layer for mu
        self.fc_sigma = nn.Linear(256 * img_size//8 * img_size//8, z_dim)  # Fully connected layer for sigma

        # Style Encoder for multiple genres
        self.style_encoder = nn.Sequential(
            nn.Linear(num_genres, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * z_dim, 256 * img_size//8 * img_size//8),
            nn.ReLU(),
            nn.Unflatten(1, (256, img_size//8, img_size//8)),  # Reshape for convolutional layers
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )

    def encode_content(self, x):
        x = self.content_encoder_conv(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

    def encode_style(self, genres):
        return self.style_encoder(genres)

    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x, genres):
        mu, sigma = self.encode_content(x)
        style_z = self.encode_style(genres)
        epsilon = torch.randn_like(sigma)
        content_z = mu + epsilon * torch.exp(sigma / 2)  # Reparametrization trick

        # Combine style and content in the latent space
        combined_z = torch.cat([content_z, style_z], dim=1)
        
        reconstructed_x = self.decode(combined_z)
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