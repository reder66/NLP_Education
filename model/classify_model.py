import torch as tc
import torch.nn as nn

class nnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_layers=2):
        super(nnModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_labels)

    def forward(self, X):
        X = tc.tensor(X, dtype=tc.float32)
        X, _ = self.lstm(X)
        X = self.fc(X[:, -1, :])
        return X
