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
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create fully conncected neural network
class Neural_Network(nn.Module):
    def __init__(self, input_size, num_classes): # input size (28 * 28 = 784)
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize network
model = Neural_Network(input_size=input_size, num_classes=num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Get to correct shape, reshapes the last sizes 1, 28, 28 to 784
        data = data.reshape(data.shape[0], -1)
        
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
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
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