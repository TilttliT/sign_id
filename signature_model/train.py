import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from .dataset import CEDARVerificationDataset
from .model import SignatureEmbeddingModel
from .utils import set_seed, save_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)
        loss = criterion(emb_anchor, emb_positive, emb_negative)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, device, target_fpr=0.05):
    model.eval()
    all_emb1, all_emb2, all_labels = [], [], []
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Validation"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
            all_labels.append(labels.cpu())

    emb1 = torch.cat(all_emb1, dim=0)
    emb2 = torch.cat(all_emb2, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    sim = torch.nn.functional.cosine_similarity(emb1, emb2).numpy()
    neg_sim = sim[labels == 0]

    neg_sim_sorted = np.sort(neg_sim)
    N = len(neg_sim)
    max_fp = int(target_fpr * N)

    if max_fp == 0:
        threshold = neg_sim_sorted[-1] + 1e-6
    else:
        threshold = neg_sim_sorted[-max_fp]

    preds = (sim >= threshold).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / len(labels)
    f1 = f1_score(labels, preds) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0

    print(
        f"Threshold: {threshold:.4f} | FPR: {fpr:.4f} (target: {target_fpr}) | TPR: {tpr:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
    )
    return threshold, acc, f1, fpr, tpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CEDAR folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--target_fpr", type=float, default=0.05, help="Target False Positive Rate for threshold find algorithm"
    )
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # train_transform = v2.Compose([
    #    v2.Resize((224, 224)),
    #    v2.RandomApply([v2.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.5),
    #    v2.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    #    v2.RandomPerspective(distortion_scale=0.05, p=0.3),
    #    v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    #    v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
    #    v2.ToImage(),
    #    v2.ToDtype(torch.float32, scale=True),
    #    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    val_transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_authors = list(range(1, 56))
    # train_authors, val_authors = train_test_split(all_authors, test_size=0.3)
    val_authors = all_authors

    # train_dataset = CEDARTripletDataset(args.data_dir, authors_list=train_authors, transform=train_transform)
    val_dataset = CEDARVerificationDataset(
        args.data_dir, authors_list=val_authors, transform=val_transform, pairs_per_author=5
    )

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SignatureEmbeddingModel()
    model = model.to(args.device)

    # criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_tpr = 0.0
    best_thresh = 0.5
    for epoch in range(1, args.epochs + 1):
        # train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        train_loss = 0
        thresh, acc, f1, fpr, tpr = validate(model, val_loader, args.device, target_fpr=args.target_fpr)
        # scheduler.step()

        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f} | TPR: {tpr:.4f}"
        )

        if fpr <= args.target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_thresh = thresh
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_thresh": best_thresh,
                    "target_fpr": args.target_fpr,
                    "tpr": best_tpr,
                },
                os.path.join(args.save_dir, "signature_model.pth"),
            )
            print(f"  -> New best model saved with TPR={tpr:.4f} at FPR={fpr:.4f}")

    print(f"Training finished. Best threshold: {best_thresh:.4f}, Best TPR: {best_tpr:.4f}")


if __name__ == "__main__":
    main()
