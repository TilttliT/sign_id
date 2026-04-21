import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class CEDARVerificationDataset(Dataset):
    def __init__(self, root_dir: str, authors_list: list, transform=None, pairs_per_author: int = 10):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []

        org_dir = os.path.join(root_dir, "full_org")

        authors_files = defaultdict(list)
        for fname in os.listdir(org_dir):
            if fname.startswith("original_"):
                parts = fname.replace("original_", "").replace(".png", "").split("_")
                author = int(parts[0])
                if author in authors_list:
                    authors_files[author].append(os.path.join(org_dir, fname))

        self.authors = list(authors_files.keys())

        for author in self.authors:
            org_list = authors_files[author]
            if len(org_list) < 2:
                continue

            for _ in range(pairs_per_author):
                p1, p2 = random.sample(org_list, 2)
                self.pairs.append((p1, p2, 1))

            other_authors = [a for a in self.authors if a != author]
            for _ in range(pairs_per_author):
                p1 = random.choice(org_list)
                other_author = random.choice(other_authors)
                p2 = random.choice(authors_files[other_author])
                self.pairs.append((p1, p2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert("L").convert("RGB")
        img2 = Image.open(path2).convert("L").convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class CEDARTripletDataset(Dataset):
    def __init__(self, root_dir: str, authors_list: list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        org_dir = os.path.join(root_dir, "full_org")

        self.org_files = defaultdict(list)
        for fname in os.listdir(org_dir):
            if fname.startswith("original_"):
                parts = fname.replace("original_", "").replace(".png", "").split("_")
                author = int(parts[0])
                if author in authors_list:
                    self.org_files[author].append(os.path.join(org_dir, fname))

        self.authors = [a for a in self.org_files if len(self.org_files[a]) >= 2]

    def __len__(self):
        return len(self.authors) * 10

    def __getitem__(self, idx):
        author = random.choice(self.authors)
        anchor_path, positive_path = random.sample(self.org_files[author], 2)
        other_author = random.choice([a for a in self.authors if a != author])
        negative_path = random.choice(self.org_files[other_author])

        anchor = Image.open(anchor_path).convert("L").convert("RGB")
        positive = Image.open(positive_path).convert("L").convert("RGB")
        negative = Image.open(negative_path).convert("L").convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
