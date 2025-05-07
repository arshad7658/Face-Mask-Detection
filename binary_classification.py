import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
# import cv2
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('dataset')
normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
batch_size = 32
lr = 1e-4
epochs = 20

class CustomDataset(Dataset):
  def __init__(self,data_dir, transform):
    self.transform = transform
    
    with_mask_dir = os.path.join(data_dir, 'with_mask')
    without_mask_dir = os.path.join(data_dir, 'without_mask')
    
    with_mask_files = os.listdir(with_mask_dir)
    without_mask_files = os.listdir(without_mask_dir)

    self.labels_with_paths = []

    classes = ['without_mask', 'with_mask']

    for cls in classes:
      if cls == 'without_mask':
        for file in without_mask_files:
          self.labels_with_paths.append([0,f'{data_dir}/without_mask/{file}'])
      if cls == 'with_mask':
        for file in with_mask_files:
          self.labels_with_paths.append([1,f'{data_dir}/with_mask/{file}'])
    # print(self.labels_with_paths[0][1])


#     # self.labels = sorted(os.listdir(data_dir))
#     # print(self.labels)
#     # self.label_nums = {cls:index for index,cls in enumerate(self.labels)}
#     # print(self.label_nums)

#     # self.images = []
#     # for file_path in :
#     #   path_to_class = os.path.join(data_dir, label)
#     #   for img_names in os.listdir(path_to_class):
#     #     img_path = os.path.join(path_to_class, img_names)
#     #     self.images.append([img_path, self.label_nums[label]])

  def __len__(self):
    return len(self.labels_with_paths)

  def __getitem__(self, index):
    cls, img_path = self.labels_with_paths[index]
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(img_path).convert('RGB')
    # img.show()
    if self.transform:
      img = self.transform(img)
    return img, torch.tensor(cls, dtype = torch.float32).unsqueeze(0)

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

train_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=40),
    transforms.ToTensor(),
    normalization
])

validation_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalization
])

test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalization
])

# #dataset setup not done yet uWu
train_dataset = CustomDataset(train_dir, train_transformations)
val_dataset = CustomDataset(val_dir, validation_transformations)
test_dataset = CustomDataset(test_dir, test_transformations)

# ##just checking
# # img, cls = test_dataset.__getitem__(330)
# # print(cls)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model = models.vgg16(pretrained = True)

for param in model.features.parameters():
  param.requires_grad = False

num_features_in = model.classifier[6].in_features
model.classifier[6]=nn.Sequential(
    nn.Linear(num_features_in, 256),
    nn.ReLU(inplace = True),
    nn.Dropout(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
)

model.to(device)

loss_function = nn.BCELoss() 

optimiser = torch.optim.Adam(params=model.parameters(), lr =lr)

best_val_loss = float('inf')

train_loss_per_epoch, val_loss_per_epoch, train_acc_per_epoch, val_acc_per_epoch = [], [], [], []

for epoch in range(epochs):
  print(f'Starting Epoch : {epoch+1}')
  model.train()

  running_train_loss, running_train_accuracy = 0.0,0.0
  total_samples = 0
  for x, y in train_loader:
    x = x.to(device)
    y =y.to(device).float()
    
    optimiser.zero_grad()
    y_hat = model(x).squeeze(1)
    loss = loss_function(y_hat, y.squeeze(1))
    loss.backward()
    optimiser.step()
    running_train_loss +=loss.item()
    predictions = torch.round(y_hat)

    running_train_accuracy+=(predictions == y.squeeze(1)).sum().item()
    total_samples += y.size(0)
  epoch_train_loss = running_train_loss/len(train_loader)
  epoch_train_accuracy = running_train_accuracy/total_samples

  model.eval()

  running_val_loss = 0.0
  running_val_acc,running_val_total_samples = 0.0,0
  with torch.no_grad():
    for x,y in val_loader:
      x = x.to(device)
      y = y.to(device).float()
      y_hat = model(x).squeeze(1)
      loss = loss_function(y_hat, y.squeeze(1))
      running_val_loss+=loss.item()
      val_predictions = torch.round(y_hat)
      running_val_acc+=(val_predictions==y.squeeze(1)).sum().item()
      running_val_total_samples += y.size(0)
    val_epoch_loss = running_val_loss/len(val_loader)
    val_epoch_accuracy = running_val_acc/running_val_total_samples

  print(f'epoch : {epoch+1}/{epochs}')
  print(f'Train Loss : {epoch_train_loss}, Train Accuracy : {epoch_train_accuracy}')
  print(f'Validation Loss:{val_epoch_loss} ,Validation Accuracy : {val_epoch_accuracy}')

  train_loss_per_epoch.append(epoch_train_loss)
  val_loss_per_epoch.append(val_epoch_loss)
  train_acc_per_epoch.append(epoch_train_accuracy)
  val_acc_per_epoch.append(val_epoch_accuracy)

torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))


model.eval()

all_predictions, all_ys = [], []
test_loss = 0.0
test_acc = 0
test_total_samples = 0

with torch.no_grad():
  test_correct_pred, test_total_samples = 0,0
  for test_x,test_y in test_loader:
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    test_y_hat = model(test_x).squeeze(1)
    loss = loss_function(test_y_hat, test_y.squeeze(1))
    test_loss+=loss.item()

    test_prediction = torch.round(test_y_hat)
    test_y = test_y.squeeze(1)
    test_correct_pred += (test_prediction==test_y).sum().item()
    test_total_samples+=test_y.size(0)

    all_predictions.extend(test_prediction.cpu().numpy())
    all_ys.extend(test_y.cpu().numpy())

test_loss = test_loss/len(test_loader)
test_acc = test_correct_pred / test_total_samples

print(f'Test Loss : {test_loss}, Test Accuracy : {test_acc}')

eps = np.arange(1, len(train_loss_per_epoch)+1)
fig, axes = plt.subplots(1,2, figsize=(10,8))
axes = axes.flat
axes[0].plot(epochs, [v.item() for v in train_loss_per_epoch], 'bo')
axes[1].plot(epochs, [v.item() for v in val_loss_per_epoch], 'ro')
axes[0].set_xlabel('Epochs')
axes[1].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[1].set_ylabel('Val Loss')
axes[1].set_title('Validation Loss Per Epoch')
axes[0].set_title('Train Loss Per Epoch')
plt.savefig('aaa.jpeg')
