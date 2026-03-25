import json

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import torch.optim as optim
from feature_extractor import ImageEncoder
from style_encoder_modules.data import IAMDataset_style, UkrDataset_style
from style_encoder_modules.training import (
    train_mixed,
    train_classification,
    train_triplet as train,
    Mixed_Encoder,
)



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Style Encoder")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2_100",
        help="type of cnn to use (resnet, densenet, etc.)",
    )
    parser.add_argument("--dataset", type=str, default="iam", help="dataset name")
    parser.add_argument(
        "--batch_size", type=int, default=320, help="input batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        required=False,
        help="number of training epochs",
    )
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="use of feature extractor or not"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to use for training / testing",
    )
    parser.add_argument(
        "--save_path", type=str, default="./style_models", help="path to save models"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        help="mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet",
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="random seed for train/val split"
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2, help="fraction of data for validation"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="root directory for the dataset (contains METAFILE.tsv for ukr)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.dataset == "iam":

        dataset_folder = "./iam_data/"

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_data = IAMDataset_style(
            dataset_folder,
            "train",
            "word",
            fixed_size=(64, 256),
            transforms=train_transform,
        )

        val_data = IAMDataset_style(
            dataset_folder,
            "val",
            "word",
            fixed_size=(64, 256),
            transforms=train_transform,
        )

        print(f"len train data {len(train_data)}")
        print(f"len val data {len(val_data)}")

        num_workers = min(2, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        _wr_dict_candidates = [
            "./writers_dict_train.json",
            os.path.join(dataset_folder, "writers_dict_train.json"),
        ]
        _wr_dict_path = next((p for p in _wr_dict_candidates if os.path.exists(p)), None)
        if _wr_dict_path is None:
            raise FileNotFoundError(
                "Could not find writers_dict_train.json. Tried: "
                + ", ".join(_wr_dict_candidates)
            )
        with open(_wr_dict_path) as _f:
            style_classes = len(json.load(_f))

    elif args.dataset == "ukr":

        dataset_folder = args.data_root if args.data_root else "./"

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load train and val as separate instances split at the form boundary.
        # This avoids the data leakage that occurs when random_split is applied
        # at the chunk level: adjacent chunks from the same handwritten line would
        # end up on both sides of the split.
        train_data = UkrDataset_style(
            dataset_folder,
            "train",
            "word",
            fixed_size=(64, 256),
            transforms=train_transform,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
        )

        val_data = UkrDataset_style(
            dataset_folder,
            "val",
            "word",
            fixed_size=(64, 256),
            transforms=train_transform,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
        )

        style_classes = train_data.num_writers

        print(f"len train data {len(train_data)}")
        print(f"len val data {len(val_data)}")
        print(f"style classes (num writers): {style_classes}")

        num_workers = min(2, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    else:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Use 'iam' or 'ukr'."
        )

    encoder_cls = Mixed_Encoder if args.mode == "mixed" else ImageEncoder
    print(f"Using {args.model} with {encoder_cls.__name__}")
    model = encoder_cls(
        model_name=args.model,
        num_classes=style_classes,
        pretrained=True,
        trainable=True,
    )
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    if args.pretrained == True:
        PATH = ""
        state_dict = torch.load(PATH, map_location=args.device)
        model_dict = model.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("Pretrained model loaded")

    model = model.to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == "mixed":
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        print("Using both classification and metric learning training")
        train_mixed(
            model,
            train_loader,
            val_loader,
            criterion_triplet,
            None,
            optimizer,
            scheduler,
            device,
            args,
        )
        print("finished training")

    if args.mode == "triplet":
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args,
        )
        print("finished training")

    elif args.mode == "classification":

        train_classification(
            model, train_loader, val_loader, optimizer, scheduler, device, args
        )
        print("finished training")


if __name__ == "__main__":
    main()
