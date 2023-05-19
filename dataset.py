from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import tqdm

class FruitDataset(Dataset):
    def __init__(self, data_root, transform = None):
        self.img_paths = []
        self.labels = []
        self.class_to_index = {}
        for idx, class_name in enumerate(os.listdir(data_root)):
            class_dir = os.path.join(data_root, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_paths.append(img_path)
                self.labels.append(idx)
            self.class_to_index[class_name] = idx
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloader(data_root, batch_size, shuffle = True):
    dataset = FruitDataset(data_root)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return dataloader

if __name__ == '__main__':
    data_root = './data/train'
    dataset = FruitDataset(data_root)
    dataloader = get_dataloader(data_root, 16)
    for samples, targets in tqdm(dataloader):
        print(samples.shape)
        pass