# Author: Ananya Agrawal
# Version: ver 1.0

import torch
from torch.utils.data import DataLoader
import numpy as np
#from torchsummary import summary
import os
from dataloader import CustomDataset
from convnetwork import ConvNet

import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, train_data_path, test_data_path, saved_model_dir="saved_model", batch_size=4, lr=0.001):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.saved_model_dir = saved_model_dir
        self.batch_size = batch_size
        self.lr = lr
        self.model = ConvNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_dataloader = None
        self.test_dataloader = None

    def prepare_dataset(self):
        train_data = CustomDataset(self.train_data_path)
        test_data = CustomDataset(self.test_data_path)
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

    def train_model(self, n_epochs):
        valid_loss_min = np.Inf

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0
            train_correct = 0
            train_total = 0
            valid_correct = 0
            valid_total = 0

            self.model.train()

            for data, target , filepath in self.train_dataloader:
                data, target ,filepath = data.to(self.device), target.to(self.device), filepath
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
                predicted = torch.round(output)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            self.model.eval()

            for data, target,filpath in self.test_dataloader:
                data, target ,filpath = data.to(self.device), target.to(self.device),filpath
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                predicted = torch.round(output)
                valid_total += target.size(0)
                valid_correct += (predicted == target).sum().item()

            train_loss = train_loss / len(self.train_dataloader)
            valid_loss = valid_loss / len(self.test_dataloader)
            train_accuracy = 100 * train_correct / train_total
            valid_accuracy = 100 * valid_correct / valid_total

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.2f}% \tValidation Accuracy: {:.2f}%'.format(
                epoch, train_loss, valid_loss, train_accuracy, valid_accuracy))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
               # torch.save(self.model.state_dict(), 'model.pt')
                self.save_model()
                valid_loss_min = valid_loss

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.saved_model_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.saved_model_dir, "model.pt")
        os.makedirs(self.saved_model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def eval_model(self, dataloader):
        self.model.eval()
        predictions = []
        targets = []
        filpaths=[]

        with torch.no_grad():
            for data, target , filpath in dataloader:
                data, target ,filpath = data.to(self.device), target.to(self.device), filpath
                output = self.model(data)
           
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                filpaths.append(filpath)
        
            # Calculate accuracy

        return np.concatenate(predictions), np.concatenate(targets), filpaths
    

    #Visualise the images and predictions
    def visualize_test_data(self, n_images=10):
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Track the number of images plotted
        images_plotted = 0
        
        # Iterate over test data batches
        for images, labels, file_paths in self.test_dataloader:
            # Move data to the device
            images = images.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(images)
            
            # Convert predictions and labels to numpy arrays
            predictions = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Get the actual number of images in the batch
            actual_batch_size = len(images)
            actual_n_images = min(n_images - images_plotted, actual_batch_size)
            
            # Plot images and corresponding predictions
            #fig, axes = plt.subplots(1, actual_n_images, figsize=(15, 3), dpi=200)
            fig, axes = plt.subplots(1, actual_n_images, figsize=(15, 3))
            for i in range(actual_n_images):
                image = images[i].cpu().squeeze().numpy()
                prediction = predictions[i]
                label = labels[i]
                file_path = file_paths[i]
                
                # Rescale image data to [0, 1]
                image = (image - image.min()) / (image.max() - image.min())
                
                # Plot image
                axes[i].imshow(image, cmap='gray')
                axes[i].set_title(f"Prediction: {prediction}, Label: {label}\nFile: {file_path}")
                axes[i].axis('off')
                
                # Increment the count of images plotted
                images_plotted += 1
            
            plt.tight_layout()
            plt.show()
            
            # Check if enough images have been plotted
            if images_plotted >= n_images:
                break



        