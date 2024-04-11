# Author: Ananya Agrawal
# Version: ver 1.0

from model_trainer import ModelTrainer
import pandas as pd
import numpy as np

import torch

import torchvision
from torchvision.transforms import ToTensor



trainer = ModelTrainer(train_data_path="data/train/", test_data_path="data/test/")
trainer.prepare_dataset()
trainer.train_model(n_epochs=5)

# Save the trained model
trainer.save_model()

#Load Model for Prediction
trainer.load_model()
predictions, targets ,filpaths = trainer.eval_model(dataloader=trainer.test_dataloader)

total = len(predictions)
correct = 0

total = len(predictions)
correct = np.sum(np.round(predictions) == targets)

accuracy = correct / total
print(f'Testing Accuracy: {accuracy:.2%}')
# Create a DataFrame to store the prediction
data = {"filpaths": np.array(filpaths).flatten(), "Target": np.array(targets).flatten(), "predictions": np.array(predictions).flatten()}
df = pd.DataFrame(data)
## Print the DataFrame
print(df)


#trainer.visualize_test_data()



