# Imports
import torch
import torch.nn.functional as F # Some parameterless functions (like activation functions).
import torchvision.transforms as transforms # Transformation we can perform on our datasets for augmentation.
from torchvision import datasets
from torch import optim # For optimizers like adam, SGD, etc.
from torch import nn # All neural network modules.
from torch.utils.data import DataLoader # Gives easier dataset managment by creating mini batches etc.

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create a Bi-directional LSTM
class Bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Bi_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def  forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagation
        out, _ = self.lstm(x, (h0, c0)) # For RNN and GRU, pass (x, h0) to self.rnn and self.gru
        out = self.fc(out[: , -1, :])
        return out

# Initialize network
model = Bi_LSTM(input_size, hidden_size, num_layers, num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent / adam step
        optimizer.step()
        
# Check accuracy on training and test to see how good the model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0
    model.eval()    
        
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        accuracy = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy*100:.2f}%')
    
    model.train()
    return accuracy

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)