import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Loading english tokenizer from spaCy
# Download first with: python -m spacy download en
spacy_eng = spacy.load("en")

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        # Get image, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize the vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        img_id = self.imgs[index]
        caption = self.captions[index]
        
        # Load Image
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img) # ==>>
        
        # Numericalize the caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
            
        return img, caption
    
class Vocabulary:
    def __init__(self, freq_threshold):
        # Mapping from index to string
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3:"<UNK>"}
        
        # Mapping from string to index
        self.stoi = {v: k for k, v in self.itos.items()}
        
        # Minimum frequency to include a word in the vocabulary
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        """
        Tokenizes english text using spacy and returns a list of lowercase tokens
        """
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)] # yesle return gareko kura k le use garxa?
    
    def build_vocabulary(self, sentence_list):
        """
        Builds vocabulary from a list of sentences.
        Only words that appear >= freq_threshold times are added.
        """
        frequencies = {} # counts word frequencies
        idx = 4 # starting index after the special tokens
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                    
                else:
                    frequencies[word] += 1
                
                # Add word to vocabulary once it reaches the threshold
                if frequencies[word] >= self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text):
        """
        Converts a caption (string) into a list of numerical indices.
        """
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch] # add batch dimension
        imgs = torch.cat(imgs, dim=0) # combine into a single tensor
        
        targets = [item[1] for item in batch] # list of caption tensors
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets
    
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    # Initialize the dataset
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    # Get the padding index from the vocabulary
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # Initialize DataLoader with custom collate function
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

if __name__ == "__main__":
    # Define transformations for the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor(),          # Convert images to tensors
        ]
    )

    # Get the DataLoader and dataset
    loader, dataset = get_loader(
        "flickr8k/images/",          # Root folder for images
        "flickr8k/captions.txt",     # Captions file
        transform=transform
    )

    # Iterate through the DataLoader and print shapes
    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)      # Should be [batch_size, 3, 224, 224]
        print(captions.shape)  # Should be [seq_len, batch_size]