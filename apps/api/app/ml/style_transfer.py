import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights
from PIL import Image
import torchvision.transforms as transforms

class NeuralStyleTransfer:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained VGG19 and set to evaluation mode
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()
        
        # Freeze all VGG parameters
        for param in vgg.parameters():
            param.requires_grad_(False)
            
        # We only need specific layers for Content and Style
        # VGG19 features are sequential. 
        # Content: conv4_2
        # Style: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.content_layers = ['21'] # conv4_2 is layer 21 in vgg.features
        self.style_layers = ['0', '5', '10', '19', '28'] # conv1_1 to conv5_1
        
        self.model = vgg
        
        # Mean and std for VGG19 normalization
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(-1, 1, 1)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(-1, 1, 1)

    def normalize(self, tensor):
        return (tensor - self.normalization_mean) / self.normalization_std

    def extract_features(self, image):
        """Extracts the necessary content and style features from the VGG network."""
        features = {'content': [], 'style': []}
        x = self.normalize(image)
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                features['content'].append(x)
            if name in self.style_layers:
                features['style'].append(x)
        return features

    def gram_matrix(self, tensor):
        """Calculates the Gram Matrix for the style loss."""
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def apply_style(self, base_image_tensor, style_image_tensor, alpha=1.0, beta=1e5, num_steps=300):
        """
        Applies style transfer using L-BFGS.
        Includes aggressive VRAM management for RTX 4050 (6GB).
        """
        base_image_tensor = base_image_tensor.to(self.device)
        style_image_tensor = style_image_tensor.to(self.device)
        
        # Target image starts as a clone of the base image
        target_image = base_image_tensor.clone().requires_grad_(True).to(self.device)
        
        # Precompute target content and style features (these don't change)
        with torch.no_grad():
            content_features = self.extract_features(base_image_tensor)['content']
            style_features = self.extract_features(style_image_tensor)['style']
            style_grams = [self.gram_matrix(sf) for sf in style_features]

        # Use L-BFGS optimizer. We might switch to Adam if L-BFGS is too memory heavy.
        optimizer = torch.optim.LBFGS([target_image])
        
        step = [0]
        while step[0] <= num_steps:
            def closure():
                target_image.data.clamp_(0, 1) # Keep image in valid range
                optimizer.zero_grad()
                
                target_features = self.extract_features(target_image)
                
                # Content Loss (MSE)
                content_loss = 0
                for tf, cf in zip(target_features['content'], content_features):
                    content_loss += F.mse_loss(tf, cf)
                
                # Style Loss (MSE of Gram matrices)
                style_loss = 0
                for tf, sg in zip(target_features['style'], style_grams):
                    tg = self.gram_matrix(tf)
                    style_loss += F.mse_loss(tg, sg)
                
                total_loss = alpha * content_loss + beta * style_loss
                total_loss.backward()
                
                step[0] += 1
                if step[0] % 50 == 0:
                    print(f"Step {step[0]}: Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")
                
                # Aggressive VRAM cleanup
                del target_features
                torch.cuda.empty_cache()
                
                return total_loss
                
            optimizer.step(closure)
            
        target_image.data.clamp_(0, 1)
        
        # Move back to CPU to save GPU memory
        final_image = target_image.detach().cpu()
        
        del target_image
        del content_features
        del style_features
        del style_grams
        torch.cuda.empty_cache()
        
        return final_image
