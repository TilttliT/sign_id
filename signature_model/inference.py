import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from typing import Tuple

from .detector import detect_signature
from .model import SignatureEmbeddingModel

DETECTOR_CONFIG = {
    "threshold_block_size_ratio": 0.05,
    "threshold_c": 20,
    "morph_kernel_ratio": 0.03,
    "morph_iterations": 2,
    "min_area_ratio": 0.005,
    "max_area_ratio": 0.95,
    "padding_ratio": 0.01,
}


class SignatureVerifier:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = SignatureEmbeddingModel()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint["best_thresh"]
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        x, y, w, h = bbox
        crop = image[y : y + h, x : x + w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb).convert("L").convert("RGB")
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def get_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        tensor = self.preprocess_image(image, bbox)
        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy()
        return emb[0]

    def verify(self, img1_path: str, img2_path: str) -> Tuple[bool, float]:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            raise ValueError("Failed to load images")

        b1 = detect_signature(img1, DETECTOR_CONFIG)
        b2 = detect_signature(img2, DETECTOR_CONFIG)

        if not b1 or not b2:
            print("Warning: no signatures were recognized in the photo")
            return False, 0.0

        e1 = self.get_embedding(img1, b1)
        e2 = self.get_embedding(img2, b2)

        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        return sim >= self.threshold, sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    verifier = SignatureVerifier(args.model, args.device)
    result, sim = verifier.verify(args.img1, args.img2)
    print(f"Similarity: {sim:.4f}")
    print(f"Result: {'Same' if result else 'Different'} authors")


if __name__ == "__main__":
    main()
