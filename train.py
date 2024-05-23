import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import numpy as np
from PIL import Image

from dataloader import ColorCorrectionDataset
from loss import MyLoss
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--in_dir', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to the ground truth directory')
    parser.add_argument('--num', type=str, required=True, help='Epochs count')
    parser.add_argument('--starting_epoch', type=str, required=False, help='Starting epoch number', default='0')
    parser.add_argument('--checkpoint', type=str, required=False, help='Checkpoint path', default='')
    
    args = parser.parse_args()
    return args

def train(model, loader, criterion, optimizer, num_epochs):
    args = parse_args()
    starting_num = int(args.starting_epoch)

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {starting_num+epoch+1}, Loss: {running_loss/len(loader)}')
        torch.save(model.state_dict(), f"models/model_{starting_num+epoch+1}.pt")
        saveImage(inputs, f"results/input_{starting_num+epoch+1}.jpg")
        saveImage(targets, f"results/gt_{starting_num+epoch+1}.jpg")
        saveImage(outputs, f"results/output_{starting_num+epoch+1}.jpg")

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    unnormalized = tensor * std + mean
    return unnormalized.clamp(0, 1) 

def saveImage(tensor, name):
    tensor = unnormalize(tensor)
    output_image = tensor.squeeze().detach().cpu().numpy()  # Remove batch dim and transfer to CPU
    output_image = np.transpose(output_image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    output_image = (output_image * 255).astype(np.uint8)  # Scale back to [0, 255]

    image = Image.fromarray(output_image).convert('RGB')
    image.save(name)


def main():
    args = parse_args()
    print(f"Input directory: {args.in_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Number of epochs: {args.num}")
    print(f"Running on: {device}")

    model = UNet(n_channels=3, n_classes=3)

    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint))

    criterion = MyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) 
    ])

    train_dataset = ColorCorrectionDataset(args.in_dir, args.gt_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    train(model, train_loader, criterion, optimizer, int(args.num))


if __name__ == '__main__':
    main()
