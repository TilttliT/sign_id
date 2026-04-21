import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import SiameseNetwork
from tqdm import tqdm


class CEDARDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.org_dir = os.path.join(root_dir, "full_org")
        self.forg_dir = os.path.join(root_dir, "full_forg")

        if not os.path.exists(self.org_dir):
            raise FileNotFoundError(f"Папка не найдена: {self.org_dir}")

        all_files = os.listdir(self.org_dir)
        person_ids = []
        for f in all_files:
            if f.startswith("original_") and f.endswith(".png"):
                parts = f.split("_")
                if len(parts) >= 2:
                    person_ids.append(parts[1])

        self.person_ids = sorted(list(set(person_ids)))

        if not self.person_ids:
            raise RuntimeError("Не удалось найти файлы подписей.")

        print(f"Найдено людей: {len(self.person_ids)}")

    def __len__(self):
        return len(self.person_ids) * 50

    def __getitem__(self, index):
        person_id = random.choice(self.person_ids)
        should_get_same = random.randint(0, 1)

        try:
            if should_get_same:
                idx1, idx2 = random.sample(range(1, 25), 2)
                img1_p = os.path.join(self.org_dir, f"original_{person_id}_{idx1}.png")
                img2_p = os.path.join(self.org_dir, f"original_{person_id}_{idx2}.png")
                label = 0.0
            else:
                idx1 = random.randint(1, 24)
                idx2 = random.randint(1, 24)
                img1_p = os.path.join(self.org_dir, f"original_{person_id}_{idx1}.png")
                img2_p = os.path.join(self.forg_dir, f"forgeries_{person_id}_{idx2}.png")
                label = 1.0

            img1 = Image.open(img1_p).convert("RGB")
            img2 = Image.open(img2_p).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__(index + 1 % self.__len__())

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean(
            (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset_path = "/Users/darya/Downloads/signatures"
    dataset = CEDARDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(15):
        epoch_loss = []

        progress_bar = tqdm(loader, desc=f"Эпоха {epoch + 1}/15")

        for img1, img2, label in progress_bar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(epoch_loss))

        print(f"Завершена эпоха {epoch + 1}, средний Loss: {np.mean(epoch_loss):.4f}")

    torch.save(model.state_dict(), "signature_model.pth")
    print("Модель сохранена.")


if __name__ == "__main__":
    main()
