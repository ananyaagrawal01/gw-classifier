# Author: Ananya Agrawal
# Version: ver 1.0

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

# Dataset preparation

class CustomDataset(Dataset):
 
    def __init__(self, datadir):
        self.imgs_path = datadir
        file_list = glob.glob(self.imgs_path + "*")

        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.pt"):
                self.data.append([img_path, class_name])

        #print(self.data)
        self.classification = {"bbh" : 0, "glitch": 1}
        self.img_dim = (21, 201)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = torch.load(os.path.join(img_path))

       # print("File Path:", img_path)  # Print file path
       # print("Class Name:", class_name)  # Print class name
        
        img = img['orthosnr'].numpy()
        class_id = self.classification[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(dim=0)
        class_id = torch.tensor([class_id]).to(dtype=torch.float32)

        # Convert the file path to a numerical representation (e.g., one-hot encoding)
       # file_path_tensor = torch.zeros(len(self.data))  # Create a tensor with zeros
       # file_path_tensor[idx] = 1  # Set the index corresponding to the file path to 1

       # file_path_tensor = torch.tensor(idx)
         # Return the file path as a string
        file_path_string = img_path
        
        return img_tensor, class_id,file_path_string
    
    