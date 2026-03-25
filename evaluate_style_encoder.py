import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from feature_extractor import ImageEncoder
from style_encoder_modules.data import IAMDataset_style, UkrDataset_style
from style_encoder_modules.training import Mixed_Encoder
from style_encoder_modules.training.mixed import _split_model_output


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min(2, os.cpu_count() or 1)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if args.dataset == "ukr":
        full_ds = UkrDataset_style(
            args.data_root,
            "train",
            "word",
            fixed_size=(64, 256),
            transforms=tfm,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
        )
        num_classes = full_ds.num_writers
    else:
        full_ds = IAMDataset_style(
            args.data_root,
            "train",
            "word",
            fixed_size=(64, 256),
            transforms=tfm,
        )
        with open(os.path.join(args.data_root, "writers_dict_train.json")) as _f:
            num_classes = len(json.load(_f))

    n_val = int(len(full_ds) * args.val_fraction)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    print(f"Validation samples: {len(val_ds)}")

    if len(val_ds) == 0:
        raise RuntimeError("No validation samples.")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    encoder_cls = Mixed_Encoder if args.mode == "mixed" else ImageEncoder
    model = encoder_cls(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=False,
        trainable=True,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    triplet_criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    running_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in val_loader:
            img = batch[0].to(device)
            wid = batch[3].to(device).long()
            pos = batch[4].to(device)
            neg = batch[5].to(device)

            logits, feat_anchor = _split_model_output(model(img))
            _, feat_pos = _split_model_output(model(pos))
            _, feat_neg = _split_model_output(model(neg))

            cls_loss = F.cross_entropy(logits, wid)
            tri_loss = triplet_criterion(feat_anchor, feat_pos, feat_neg)
            loss = cls_loss + tri_loss

            running_loss += loss.item() * wid.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == wid).sum().item()
            n_total += wid.size(0)

    val_loss = running_loss / max(n_total, 1)
    val_acc = 100.0 * n_correct / max(n_total, 1)

    print(f"Total validation samples: {n_total}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset", type=str, default="iam", choices=["iam", "ukr"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model", type=str, default="mobilenetv2_100")
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "triplet", "classification"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--split_seed", type=int, default=42, help="random seed for train/val split"
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2, help="fraction of data for validation"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
