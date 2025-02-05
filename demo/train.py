import matplotlib
import torch
matplotlib.style.use('ggplot')
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

def fit(model, dataloader):
    print('train')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(dataloader.dataset)
    print(f"loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}")
    return train_loss, train_accuracy

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    train_epoch_loss, train_epoch_accuracy = fit(model, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(model, valloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    early_stopping(val_epoch_loss)

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
