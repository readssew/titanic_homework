# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler

feature_cols = [col for col in train.columns if col != 'datetime']
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# %%
# split X and y
# 以'Global_active_power'为预测目标
target_col = 'Global_active_power'
X_train = train[feature_cols].values
y_train = train[target_col].values
X_test = test[feature_cols].values
y_test = test[target_col].values

# 转换为监督学习格式（以过去24小时预测下一个小时）
def create_sequences(X, y, seq_length=24):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 24
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# %%
# creat dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# %%
# build a LSTM model
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out.squeeze()

input_size = X_train_seq.shape[2]
model = LSTMModel(input_size)

# %%
# train the model
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# %%
# evaluate the model on the test set
model.eval()
preds = []
actuals = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.extend(pred)
        actuals.extend(yb.numpy())
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actuals, preds)
print("Test MSE:", mse)


# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(actuals[:500], label='Ground Truth')
plt.plot(preds[:500], label='Prediction')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Global_active_power (normalized)')
plt.title('LSTM Prediction vs Ground Truth')
plt.show()
