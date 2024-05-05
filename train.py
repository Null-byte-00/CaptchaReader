#from create_dataset import CaptchaDataset
from model import SingleDigitModel
import torch
import pandas as pd

EPOCHS = 100
BATCH_SIZE = 32

dataset = pd.read_hdf('dataset/dataset.hdf')
dataset = dataset.sample(frac=1)
model = SingleDigitModel()

train, test = dataset[:int(len(dataset) * 0.8)], dataset[int(len(dataset) * 0.8):]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_loss(model):
    avg_loss = 0
    for i in test.index:
        x, y = test['image'][i], test['label'][i]
        x , y = torch.tensor(x), torch.tensor(y)
        target = torch.zeros(10)
        target[y] = 1
        loss = model.criterion(model.forward(x), target)
        avg_loss += loss
    avg_loss /= len(test)
    return avg_loss


for epoch in range(EPOCHS):
    avg_loss = 0
    for i in train.index:
        x, y = train['image'][i], train['label'][i]
        x , y = torch.tensor(x), torch.tensor(y)
        target = torch.zeros(10)
        target[y] = 1
        loss = model.train_datapoint(x, target, verbose=False)
        avg_loss += loss
        if i % BATCH_SIZE == 0:
            model.end_batch()
    avg_loss /= len(train)
    test_avg_loss = test_loss(model)
    print(f"Epoch: {epoch}, Loss: {avg_loss}, test loss: {test_avg_loss}")
    torch.save(model.state_dict(), 'models/model.pth')

torch.save(model.state_dict(), 'models/model.pth')