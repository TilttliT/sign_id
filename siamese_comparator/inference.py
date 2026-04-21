import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from .model import SiameseNetwork


class SignatureComparator:
    def __init__(self, model_path=None, threshold=0.7):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.threshold = threshold

        if model_path is None:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, "signature_model.pth")

        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def compare(self, img1_path, img2_path):
        """Возвращает (True/False, дистанция)"""
        img1 = self.preprocess(img1_path)
        img2 = self.preprocess(img2_path)

        with torch.no_grad():
            out1, out2 = self.model(img1, img2)
            distance = F.pairwise_distance(out1, out2).item()

        is_same = distance < self.threshold
        return is_same, distance


def check_signatures(path1, path2):
    comparator = SignatureComparator()
    return comparator.compare(path1, path2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    args = parser.parse_args()

    comp = SignatureComparator()
    result, dist = comp.compare(args.img1, args.img2)
    print(f"Результат: {result}, Дистанция: {dist:.4f}")
