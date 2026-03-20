import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from feature_extractor import ImageEncoder
from style_encoder_modules.data import IAMDataset_style
from style_encoder_modules.training.mixed import _split_model_output


def _remap_val_to_train_label_space(train_ds, val_ds):
    train_map = train_ds.writer_id_to_index
    remapped_val_data = []
    skipped_unseen = 0

    for (img, transcr, _, img_path), raw_writer_id in zip(val_ds.data, val_ds.initial_writer_ids):
        if raw_writer_id in train_map:
            remapped_label = int(train_map[raw_writer_id])
            remapped_val_data.append((img, transcr, remapped_label, img_path))
        else:
            skipped_unseen += 1

    val_ds.data = remapped_val_data
    return skipped_unseen


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min(2, os.cpu_count() or 1)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_ds = IAMDataset_style(
        args.data_root,
        "train",
        "word",
        fixed_size=(64, 256),
        transforms=tfm,
    )
    val_ds = IAMDataset_style(
        args.data_root,
        "val",
        "word",
        fixed_size=(64, 256),
        transforms=tfm,
    )

    skipped_unseen = _remap_val_to_train_label_space(train_ds, val_ds)
    print(f"Validation samples kept: {len(val_ds)}")
    print(f"Skipped unseen writer samples: {skipped_unseen}")

    if len(val_ds) == 0:
        raise RuntimeError("No validation samples left after label remapping.")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    max_label = max(sample[2] for sample in train_ds.data) if len(train_ds.data) > 0 else -1
    num_classes = max(train_ds.wclasses, int(max_label) + 1)

    model = ImageEncoder(
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
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model", type=str, default="mobilenetv2_100")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()