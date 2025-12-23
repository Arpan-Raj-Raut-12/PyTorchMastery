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
input_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True

# Load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create fully conncected neural network
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution(this config does not change the output size i.e 28 * 28 input and 28 * 28 output))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(in_features= 16*7*7, out_features=num_classes)
            
    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    

# Initialize network
model = CNN(input_channels=input_channels, num_classes=num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loading the model
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train network
for epoch in range(num_epochs):
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
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
            x = x.to(device=device)
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

checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
save_checkpoint(checkpoint)