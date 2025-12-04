import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.dynamic_net import Vcnet
from utils.eval import curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset_from_matrix(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def adjust_learning_rate(optimizer, init_lr, epoch, lr_type='fixed', num_epoch=800):
    if lr_type == 'cos':
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        lr = init_lr * (0.96 ** epoch)
    else:
        lr = init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def criterion(out, y, alpha=0.5, epsilon=1e-5):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()


if __name__ == "__main__":
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    num_epoch = 500
    verbose = 100

    file_path = '/Users/liushucheng/Desktop/Weightednet/dataset/CMR/County_annual_PM25_CMR.csv'
    data = pd.read_csv(file_path)
    data_2020 = data[data['Year'] == 2000]
    t = data_2020.iloc[:, -2].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    t_normalized = scaler.fit_transform(t.reshape(-1, 1)).flatten()
    t = t_normalized
    y = data_2020.iloc[:, -1].values

    file_path = '/Users/liushucheng/Desktop/Weightednet/dataset/CMR/County_RAW_variables.csv'
    df = pd.read_csv(file_path)
    columns_2000 = [col for col in df.columns if '2000' in col]
    df_normalized = df[columns_2000].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    x = df_normalized.values
    y = y / 1000

    data_matrix = np.column_stack((t, x, y))

    idx = list(range(2132))
    random.seed(11)
    random.shuffle(idx)
    train_idx = idx[:1500]
    test_idx = idx[1500:]

    train_matrix = torch.tensor(data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(data_matrix[test_idx, :], dtype=torch.float32)

    second_row = np.arange(0, 1, 0.01)
    first_row = np.arange(1, len(second_row) + 1)
    t_grid = torch.tensor(np.vstack([second_row, first_row]))

    train_loader = get_iter(
        train_matrix, batch_size=len(train_matrix), shuffle=False)
    test_loader = get_iter(
        test_matrix, batch_size=len(test_matrix), shuffle=False)

    cfg_density = [(8, 8, 1, 'relu'), (8, 8, 1, 'tanh')]
    num_grid = 20
    cfg = [(8, 4, 1, 'tanh'), (4, 1, 1, 'id')]
    degree = 4
    knots = [0.3, 0.7]

    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model = model.to(device)
    model._initialize_weights()

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=0.0005, momentum=momentum, weight_decay=wd, nesterov=True)

    for epoch in range(num_epoch):
        for inputs, y_batch in train_loader:
            t_input = inputs[:, 0]
            x_input = inputs[:, 1:]
            optimizer.zero_grad()
            out = model.forward(t_input, x_input)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % verbose == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    t_grid_hat, mse = curve(model, test_matrix, t_grid)

    t_vals = t_grid[0].detach().cpu().numpy()
    t_denormalized = scaler.inverse_transform(t_vals.reshape(-1, 1)).flatten()

    plt.figure()
    t_original = scaler.inverse_transform(t.reshape(-1, 1)).flatten()
    plt.scatter(t_original, y, color='red', s=5)
    plt.xlabel('PM2.5 Concentration (µg/m$^3$)')
    plt.ylabel('Cardiovascular Mortality Rate (per 100K)')
    plt.title('Observed PM2.5 vs. Mortality Rate Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
