from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import tqdm
from torchvision import transforms

def class_to_index(data_root):
    class_to_index = {}
    for idx, class_name in enumerate(os.listdir(data_root)):
        class_to_index[class_name] = idx
    return class_to_index


class CustomDataset(Dataset):
    def __init__(self, data_root, mapping, transform = None):
        self.transform = transform
        self.img_paths = []
        self.targets = []
        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_paths.append(img_path)
                self.targets.append(mapping[class_name])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def get_dataloader(dataset, batch_size, shuffle = True):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return dataloader

if __name__ == '__main__':
    data_root = './data/train'
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = CustomDataset(data_root, transform=transform_train)
    dataloader = get_dataloader(dataset, 16)
    for samples, targets in tqdm.tqdm(dataloader, total = len(dataloader)):
        pass