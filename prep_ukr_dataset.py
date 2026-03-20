import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_metafile(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise RuntimeError(f"Empty metafile: {path}")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            filename, transcription = parts
            rows.append((filename.strip(), transcription.strip()))
    if not rows:
        raise RuntimeError(f"No usable rows in {path}")
    return rows


def build_writer_from_filename(filename: str):
    stem = Path(filename).stem
    return stem.split("-")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Convert Ukrainian lines dataset to IAM-compatible format for style encoder"
    )
    parser.add_argument("--metafile", type=str, default="./METAFILE.tsv")
    parser.add_argument("--lines_dir", type=str, default="./lines/lines")
    parser.add_argument("--out_root", type=str, default="./iam_data")
    parser.add_argument("--split_dir", type=str, default="./utils/aachen_iam_split")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metafile = Path(args.metafile)
    lines_dir = Path(args.lines_dir)
    out_root = Path(args.out_root)
    split_dir = Path(args.split_dir)

    if not metafile.exists():
        raise FileNotFoundError(f"Missing metafile: {metafile}")
    if not lines_dir.exists():
        raise FileNotFoundError(f"Missing lines directory: {lines_dir}")

    ascii_dir = out_root / "ascii"
    words_root = out_root / "words"

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    ascii_dir.mkdir(parents=True, exist_ok=True)
    words_root.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metafile(metafile)
    forms_map = {}
    words_lines = []
    copied = 0
    missing_images = 0

    for filename, transcription in rows:
        stem = Path(filename).stem
        name_parts = stem.split("-")
        if len(name_parts) < 4:
            continue

        form_id = "-".join(name_parts[:2])
        writer_id = build_writer_from_filename(filename)

        src_img = lines_dir / filename
        if not src_img.exists():
            missing_images += 1
            continue

        # Expected by IAMDataset_style loader:
        # root/words/<p0>/<p0-p1>/<full_name>.png
        p0 = name_parts[0]
        p01 = "-".join(name_parts[:2])
        dst_img = words_root / p0 / p01 / f"{stem}.png"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        copied += 1

        forms_map.setdefault(form_id, writer_id)
        if not transcription:
            transcription = "sample"
        # IAM words.txt compatibility: transcription starts at token index 8
        words_lines.append(f"{stem} ok 0 0 0 0 0 0 {transcription}")

    if copied == 0:
        raise RuntimeError(
            "No images copied. Check --lines_dir and metafile filenames."
        )

    forms_path = ascii_dir / "forms.txt"
    words_path = ascii_dir / "words.txt"
    forms_path.write_text(
        "\n".join(f"{k} {v}" for k, v in sorted(forms_map.items())),
        encoding="utf-8",
    )
    words_path.write_text("\n".join(words_lines), encoding="utf-8")

    form_ids = np.array(sorted(forms_map.keys()))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(form_ids))

    n_test = int(len(form_ids) * args.test_fraction)
    n_val = int(len(form_ids) * args.val_fraction)
    n_train = len(form_ids) - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Invalid split fractions produce empty train split.")

    train_ids = form_ids[perm[:n_train]]
    val_ids = form_ids[perm[n_train : n_train + n_val]]
    test_ids = form_ids[perm[n_train + n_val :]]

    (split_dir / "train_val.uttlist").write_text(
        "\n".join(train_ids.tolist()), encoding="utf-8"
    )
    (split_dir / "validation.uttlist").write_text(
        "\n".join(val_ids.tolist()), encoding="utf-8"
    )
    (split_dir / "test.uttlist").write_text(
        "\n".join(test_ids.tolist()), encoding="utf-8"
    )

    def writer_map(ids):
        writers = sorted({forms_map[fid] for fid in ids})
        return {str(w): i for i, w in enumerate(writers)}

    Path("writers_dict_train.json").write_text(
        json.dumps(writer_map(train_ids), ensure_ascii=False),
        encoding="utf-8",
    )
    Path("writers_dict_val.json").write_text(
        json.dumps(writer_map(val_ids), ensure_ascii=False),
        encoding="utf-8",
    )
    Path("writers_dict_test.json").write_text(
        json.dumps(writer_map(test_ids), ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Rows in metafile: {len(rows)}")
    print(f"Images copied: {copied}")
    print(f"Missing images: {missing_images}")
    print(f"Forms total: {len(form_ids)}")
    print(f"Train/Val/Test forms: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Wrote: {forms_path}")
    print(f"Wrote: {words_path}")
    print(f"Wrote split files in: {split_dir}")
    print(
        "Wrote: writers_dict_train.json, writers_dict_val.json, writers_dict_test.json"
    )


if __name__ == "__main__":
    main()
