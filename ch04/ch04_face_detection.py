# pip install facenet-pytorch

from facenet_pytorch import MTCNN
import torch
from PIL import Image

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create an MTCNN object
mtcnn = MTCNN(keep_all=True, device=device)

# Load an image
image = Image.open('path_to_your_image.jpg')

# Perform face detection
boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

# Print the bounding boxes, probabilities of the detected faces, and the landmarks
for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
    print(f'Face {i+1}:')
    print(f'Bounding box: {box}')
    print(f'Probability: {prob}')
    print(f'Landmarks: {landmark}')
