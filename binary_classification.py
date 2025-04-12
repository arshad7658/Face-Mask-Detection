import torch
from torch.utils.data import Dataset, DataLoader
import os
# import cv2
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('datasets')
normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
batch_size = 32
lr = 1e-2
epochs = 20

class CustomDataset(Dataset):
  def __init__(self,data_dir, transform):
    self.data_dir = data_dir
    self.transform = transform
    self.labels = sorted(os.listdir(self.data_dir))
    self.label_nums = {cls:index for index,cls in enumerate(self.labels)}
    self.images = []
    for label in self.labels:
      path_to_class = os.path.join(data_dir, label)
      for img_names in os.listdir(path_to_class):
        img_path = os.path.join(path_to_class, img_names)
        self.images.append([img_path, self.label_nums[label]])
  def __len__(self):
    return len(self.images)
  def __getitem__(self, index):
    img_path, cls = self.images[index]
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(img_path).convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img, torch.tensor(cls, dtype = torch.float32).unsqueeze(0)

train_dir = os.path.join(data_dir, 'train')
val_dir = train_dir = os.path.join(data_dir, 'validation')
test_dir = train_dir = os.path.join(data_dir, 'test')

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

#dataset setup not done yet uWu
train_dataset = CustomDataset(train_dir, train_transformations)
val_dataset = CustomDataset(val_dir, validation_transformations)
test_dataset = CustomDataset(test_dir, test_transformations)


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model = models.vgg16(pretrained = True)

for param in model.features.parameters():
  param.requires_grad = False

num_features_in = model.classifier[6].in_features
model.classifier[6]=nn.Sequential(
    nn.Linear(num_features_in, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
)

model.to(device)

loss_function = nn.BCELoss()
optimiser = torch.optim.Adam(params=model.parameters(), lr =lr)

best_val_loss = float('inf')
epoch_no_improvement = 0
when_to_stop = 5

for epoch in range(epochs):

  model.train()

  runnin_loss = 0.0
  correct_predictions, total_samples = 0,0
  for x, y in train_loader:
    x = x.to(device)
    y =y.to(device)
    
    optimiser.zero_grad()
    y_hat = model(x)
    loss = loss_function(y_hat, y)
    loss.backward()
    optimiser.step()
    runnin_loss +=loss.item() * x.size(0)
    predictions = torch.round(y_hat)
    predictions = predictions.squeeze(1)
    y=y.squeeze(1)

    correct_predictions+=(predictions == y).sum().item()
    total_samples += y.size(0)
  epoch_loss = runnin_loss/total_samples
  epoch_accuracy = correct_predictions/total_samples

  model.eval()

  val_loss = 0.0
  val_acc,val_total_samples = 0,0
  with torch.no_grad():
    for x,y in val_loader:
      x = x.to(device)
      y = y.to(device)
      y_hat = model(x)
      loss = loss_function(y_hat, y)
      val_loss+=loss.item()*x.size(0)
      val_predictions = torch.round(y_hat)
      val_acc+=(val_predictions==y).sum().item()
      val_total_samples += y.size(0)
    val_epoch_loss = val_loss/val_total_samples
    val_epoch_accuracy = val_acc/val_total_samples

  print(f'epoch : {epoch+1/epochs}')
  print(f'Train Loss : {epoch_loss}, Train Accuracy : {epoch_accuracy}')
  print(f'Validation Loss:{val_epoch_loss} ,Validation Accuracy : {val_acc}')

  if loss < best_val_loss:
    best_val_loss = loss
    epoch_no_improvement = 0
    torch.save(model.state_dict(), 'best_model_so_far.pth')

  else:
    epoch_no_improvement+=1
    if epoch_no_improvement==when_to_stop:
      print('overfitting. Better to stop training')
      model.load_state_dict(torch.load('best_model_so_far.pth'))
      break

model.load_state_dict(torch.load('best_model_so_far.pth'))
model.eval()
all_predictions, all_ys = [], []
test_loss = 0.0
test_acc = 0
test_total_samples = 0

with torch.no_grad():
  for test_x,test_y in test_loader:
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    test_y_hat = model(test_x)
    loss = loss_function(test_y_hat, test_y)
    test_loss+=loss.item()*test_x.size(0)

    test_prediction = torch.round(test_y_hat)
    test_y = test_y.squeeze(0)
    test_correct_pred = (test_prediction==test_y).sum().item()
    test_total_samples+=1

    all_predictions.extend(test_prediction.cpu().numpy())
    all_ys.extend(test_y.cpu().numpy())

test_loss = test_loss/test_total_samples
test_acc = test_correct_pred / test_total_samples

print(f'Test Loss : {test_loss}, Test Accuracy : {test_acc}')

cm = confusion_matrix(all_ys, all_predictions)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = train_dataset.labels,
            yticklabels = train_dataset.labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('final.jpeg')

path = os.path.join(data_dir,'test','with_mask','with_mask_7.jpg')
class_names = train_dataset.labels
img = Image.open(path)
img = test_transformations(img).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
  output = model(img)
  prob = torch.sigmoid(output).item()
  cls = class_names[round(prob)]
  print(cls)
