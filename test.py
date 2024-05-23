import argparse
import torch
from model import UNet
import numpy as np
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--input', type=str, required=True, help='Path to the ground truth directory')
    
    args = parser.parse_args()
    return args

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    unnormalized = tensor * std + mean
    return unnormalized.clamp(0, 1) 

def saveImage(tensor, name):
    tensor = unnormalize(tensor)
    output_image = tensor.squeeze().detach().cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = (output_image * 255).astype(np.uint8) 

    image = Image.fromarray(output_image).convert('RGB')
    image.save(name)

def main():
    args = parse_args()

    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(args.model_path))

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) 
    ])

    input_image = Image.open(args.input).convert('RGB')
    input_tensor = transform(input_image).to(device).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    saveImage(output_tensor, 'output.jpg')

if __name__ == '__main__':
    main()