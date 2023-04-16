#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import time
import os
import pandas as pd 
import io
try:
    import smdebug.pytorch as smd
except ModuleNotFoundError:
    print("module 'smdebug' is not installed. Probably an inference container")
    
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True #disable image truncated error

import argparse

class GTSRBDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'Path'])
        image = Image.open(img_path)
        label = self.img_labels. loc[idx, 'ClassId']
        if self.transform:
            image = self.transform(image)
        return image, label

def test(model, test_loader, criterion, device, hook):
    hook.set_mode(smd.modes.EVAL)
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    print(f"Test set: Average loss: {total_loss}, Accuracy: {100 * total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, device, hook, num_epochs):
    epochs = num_epochs
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Train set: [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) Time: {}".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0 * accuracy,
                        time.asctime()
                    )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

        if loss_counter == 1:
            break
    return model
    
def net(num_classes):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  


    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, num_classes))
    return model

def create_data_loaders(data, labels_file, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    data = GTSRBDataset(annotations_file = labels_file, img_dir=data, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                shuffle=True)
    return data_loader

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def main(args):
    os.environ["SM_CHANNEL_TRAIN"]
    os.environ["SM_CHANNEL_VALID"]
    os.environ["SM_CHANNEL_TEST"]
    # Create loaders
    train_loader = create_data_loaders(os.environ["SM_CHANNEL_TRAIN"], os.environ["SM_CHANNEL_TRAIN"] +  "/gt_" + str(args.num_of_classes) + ".csv", args.batch_size)
    validation_loader = create_data_loaders(os.environ["SM_CHANNEL_VALID"], os.environ["SM_CHANNEL_VALID"] + "/gt_" + str(args.num_of_classes) + ".csv", args.batch_size)
    test_loader = create_data_loaders(os.environ["SM_CHANNEL_TEST"], os.environ["SM_CHANNEL_TEST"] +  "/gt_" + str(args.num_of_classes) + ".csv", args.batch_size)
    '''
    Initialize a model by calling the net function
    '''
    model=net(args.num_of_classes)
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    '''
    Create loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion)
    optimizer = optim.Adadelta(model.fc.parameters(), lr=args.lr)
    
    '''
    Call the train function to start training model
    '''
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook, num_epochs = args.num_epochs)
    
    '''
    Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    Save the trained model
    '''
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify all the hyperparameters
    '''
    parser.add_argument(
        "--lr", type = float, metavar="LR", help="learning rate (default: 0.01)" 
    )
    parser.add_argument(
        "--batch-size", type = int, metavar="N",
        help="input batch size for training"
    )

    parser.add_argument(
        "--num-epochs", type = int, metavar="E", help="Number of epochs"
    )

    parser.add_argument(
        "--num-of-classes", type = int, metavar="NUM", help="Number of output classes"
    )
    
    parser.add_argument("--model_dir", type = str, default=os.environ["SM_MODEL_DIR"])
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    args=parser.parse_args()
    
    main(args)
