from model import SingleDigitModel
from process import image_to_tensor
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SingleDigitModel()
model = model.to(device)
model.load('models/model.pth')

if len(sys.argv) < 2:
    print("Please provide an image path")
    sys.exit(1)

image = image_to_tensor(sys.argv[1])
image = image.to(device)

print(model.forward(image).argmax().item())
