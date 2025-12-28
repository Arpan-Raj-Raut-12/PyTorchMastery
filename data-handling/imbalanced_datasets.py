import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
import os

# Methods for dealing with imbalanced datasets:
# 1. Oversampling
# 2. Class weighting

def get_loader(root_dir, batch_size):
# Basic transforms: rezie and converting to tensor
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    
    subdirectories = dataset.classes
    class_weights = []
    
    # loop through each subdirectory and calculate the class weight
    # that is 1 / len(files) in that subdirectory
    for subdir in subdirectories:
        files = os.listdir(os.path.join(root_dir, subdir))
        class_weights.append(1 / len(files))
    
    sample_weights = [0] * len(dataset)
    
    for idx, (data, label) in enumerate(dataset):
    # Get the class weight for this sample's label
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader
    
def main():
    loader = get_loader(root_dir="../data/imbalanced_classes_dataset/", batch_size=8)
    
    num_retrievers = 0
    num_elkhounds = 0
    
    for epoch in range(10):
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkhounds += torch.sum(labels == 1)
            
    print(num_retrievers.item())
    print(num_elkhounds.item())
    
if __name__=="__main__":
    main()