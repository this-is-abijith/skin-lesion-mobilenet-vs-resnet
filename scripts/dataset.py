import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dirs = img_dirs  # list of folder paths to search
        self.transform = transform

        # build image_id -> full path map once (fast lookup)
        self.path_map = {}
        for d in img_dirs:
            for fname in os.listdir(d):
                if fname.endswith('.jpg'):
                    img_id = fname.replace('.jpg', '')
                    self.path_map[img_id] = os.path.join(d, fname)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.path_map[row['image_id']]
        image = Image.open(img_path).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label