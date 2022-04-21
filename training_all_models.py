import os, sys
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.io import read_image
from torch import nn
from collections import OrderedDict
from torch import optim
from torchvision.datasets import ImageFolder
import argparse


def run_train_all_models(utkface_data_dir, ckplus_data_dir, base_dir, n_epochs=1):
    n_epochs = n_epochs
    # Crop the face part from the data_sample using MTCNN  and save them
    mtcnn = MTCNN(margin=30)
    path_to_sample_folder = utkface_data_dir
    path_to_cropped_folder = os.path.join(base_dir, 'UTKFace_cropped')
    if not os.path.isdir(path_to_cropped_folder): os.mkdir(path_to_cropped_folder)

    list_image_names = os.listdir(path_to_sample_folder)

    for name in list_image_names:
        try:
            for_crop_img = Image.open(os.path.join(path_to_sample_folder, name))
            boxes, probs = mtcnn.detect(for_crop_img)
            x, y, x2, y2 = [int(x) for x in boxes[0]]
            cropped_img = for_crop_img.crop((x, y, x2, y2))
            # save cropped image
            cropped_img.save(os.path.join(path_to_cropped_folder, name))
            print('saving cropped image :'+name)
        except Exception as e:
            print(e)
            pass

    # General transform
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Train for gender model
    ########################################################################

    data_dir = path_to_cropped_folder
    train_data = GenderModelDataset(data_dir, data_transform, model_name='gender')
    train_size = int(len(os.listdir(data_dir)) * 0.8)
    val_size = len(os.listdir(data_dir)) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 100)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(100, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # fc params require grad True
    model.fc = fc
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # Train
    checkpoint_path = os.path.join(base_dir, 'resnet_18_gender_model.pt')
    train(model, optimizer, criterion, train_dataloader, val_dataloader, checkpoint_path, n_epochs=n_epochs)

    ########################################################################

    # Train for age model
    ########################################################################
    # create model
    resnext50 = models.resnext50_32x4d(pretrained=True)
    for param in resnext50.parameters():
        param.requires_grad = False
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 117)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    resnext50.fc = fc
    # create dataloader
    train_data = GenderModelDataset(data_dir, data_transform, model_name='age')
    train_size = int(len(os.listdir(data_dir)) * 0.8)
    val_size = len(os.listdir(data_dir)) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    # start Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnext50.fc.parameters(), lr=0.001)
    checkpoint_path = os.path.join(base_dir, 'resnext50_age_model.pt')
    train(resnext50, optimizer, criterion, train_dataloader, val_dataloader, checkpoint_path, n_epochs=n_epochs)

    ########################################################################

    # Train for emotion model
    ########################################################################

    resnext50_2 = models.resnext50_32x4d(pretrained=True)
    for param in resnext50_2.parameters():
        param.requires_grad = False
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 7)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    resnext50_2.fc = fc

    training_data = ImageFolder(
        root=ckplus_data_dir, transform=data_transform)

    train_size = int(len(training_data)*0.8)
    val_size = len(training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(training_data, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(resnext50_2.fc.parameters(), lr=0.001)
    checkpoint_path = os.path.join(base_dir, 'resnext50_emotions_model.pt')
    train(resnext50_2, optimizer, criterion, train_dataloader, val_dataloader, checkpoint_path, n_epochs=n_epochs)

    ########################################################################

# Main training loop
def train(model, optimizer,criterion, trainloader, validloader, checkpoint_path, n_epochs=1):
    min_val_loss = np.Inf
    n_epochs = n_epochs
    # Main loop
    for epoch in range(n_epochs):
        # Initialize validation loss for epoch
        val_loss = 0
        running_loss = 0
        model.train()
        # Training loop
        for my_loader in trainloader:
            try:
                data = my_loader['image']
                targets = my_loader['target']
            except:
                data = my_loader[0]
                targets = my_loader[1]
            optimizer.zero_grad()
            # Generate predictions
            out = model(data)
            # Calculate loss
            loss = criterion(out, targets)
            running_loss += loss
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            print('batch loss: ', loss)
        print("Training Loss: {:.6f}".format(running_loss/len(trainloader)))
        # Validation loop
        with torch.no_grad():
            model.eval()
            for my_loader in validloader:
                try:
                    data = my_loader['image']
                    targets = my_loader['target']
                except:
                    data = my_loader[0]
                    targets = my_loader[1]
                # Generate predictions
                out = model(data)
                # Calculate loss
                loss = criterion(out, targets)
                val_loss += loss
            print("validation Loss: {:.6f}".format(val_loss / len(validloader)))
            # Average validation loss
            val_loss = val_loss / len(validloader)
            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model.state_dict(), checkpoint_path)
                min_val_loss = val_loss


# AGE and GENDER dataset construct
class GenderModelDataset(Dataset):
    def __init__(self, data_dir, transform, model_name):
        self.transform = transform
        self.data_dir = data_dir
        self.model_name = model_name

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        list_all_names = os.listdir(self.data_dir)
        image_dir_idx = os.path.join(self.data_dir, list_all_names[idx])
        image = Image.open(image_dir_idx)
        if self.model_name == 'gender':
            target = list_all_names[idx].split('_')[1]
            if target not in ['0', '1']: target = 1  # error in sample
            target = torch.tensor(int(target))
        if self.model_name == 'age':
            target = list_all_names[idx].split('_')[0]
            target = torch.tensor(int(target))
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'target': target}
        return sample


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    #print(base_dir)
    parser = argparse.ArgumentParser(description="usage info")

    parser.add_argument('-ckplus_data_dir', '--ckplus_data_dir', help="CKPLUS data full directory", type=str, required=True)
    parser.add_argument('-utkface_data_dir', '--utkface_data_dir', help="UTKFace data full directory", type=str, required=True)
    parser.add_argument('-n_epochs', '--n_epochs', type=int)
    args = parser.parse_args()
    #print(args.utkface_data_dir, args.ckplus_data_dir)
    run_train_all_models(args.utkface_data_dir, args.ckplus_data_dir, base_dir, args.n_epochs)

