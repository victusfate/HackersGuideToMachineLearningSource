# pip install torch torchvision
import torch
from torchvision import models, transforms
from PIL import Image

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Define the image transformations (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform the image
image = Image.open('path_to_your_image.jpg')
image = transform(image).unsqueeze(0).to(device)

# Perform the image classification
output = model(image)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

print(f'Predicted class: {predicted_class.item()}')
