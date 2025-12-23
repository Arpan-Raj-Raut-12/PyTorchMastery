# Imports
import torch, torchvision
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
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

# Load data
train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Identity function
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# Initialize model
model = torchvision.models.vgg19(pretrained=True)

# For freezing the layers to finetune the last few layers
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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