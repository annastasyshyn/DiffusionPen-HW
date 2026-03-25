import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


CHUNK_WIDTH = 250
MIN_CHUNK_WIDTH = 100


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
    parts = stem.split("-")
    return parts[1] if len(parts) > 1 else parts[0]


def slice_line_image(img):
    """Slice a wide line image into word-sized chunks."""
    w, h = img.size
    if w <= CHUNK_WIDTH:
        return [img]
    chunks = []
    x = 0
    while x < w:
        x_end = min(x + CHUNK_WIDTH, w)
        if w - x_end < MIN_CHUNK_WIDTH and x_end < w:
            x_end = w
        chunks.append(img.crop((x, 0, x_end, h)))
        x = x_end
    return chunks


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
    total_chunks = 0
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

        img = Image.open(src_img)
        chunks = slice_line_image(img)

        p0 = name_parts[0]
        p01 = "-".join(name_parts[:2])
        dst_dir = words_root / p0 / p01
        dst_dir.mkdir(parents=True, exist_ok=True)

        forms_map.setdefault(form_id, writer_id)
        if not transcription:
            transcription = "sample"

        line_part = name_parts[3]
        for ci, chunk in enumerate(chunks):
            chunk_id = f"{line_part}{ci:02d}"
            chunk_stem = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}-{chunk_id}"
            chunk.save(dst_dir / f"{chunk_stem}.png")
            total_chunks += 1
            words_lines.append(
                f"{chunk_stem} ok 0 0 0 0 0 0 {transcription}"
            )

    if total_chunks == 0:
        raise RuntimeError(
            "No images created. Check --lines_dir and metafile filenames."
        )

    forms_path = ascii_dir / "forms.txt"
    words_path = ascii_dir / "words.txt"
    forms_path.write_text(
        "\n".join(f"{k} {v}" for k, v in sorted(forms_map.items())),
        encoding="utf-8",
    )
    words_path.write_text("\n".join(words_lines), encoding="utf-8")

    all_form_ids = np.array(sorted(forms_map.keys()))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_form_ids))

    n_test = int(len(all_form_ids) * args.test_fraction)
    n_val = int(len(all_form_ids) * args.val_fraction)
    n_train = len(all_form_ids) - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Invalid split fractions produce empty train split.")

    train_ids = all_form_ids[perm[:n_train]]
    val_ids = all_form_ids[perm[n_train : n_train + n_val]]
    test_ids = all_form_ids[perm[n_train + n_val :]]

    (split_dir / "train_val.uttlist").write_text(
        "\n".join(train_ids.tolist()), encoding="utf-8"
    )
    (split_dir / "validation.uttlist").write_text(
        "\n".join(val_ids.tolist()), encoding="utf-8"
    )
    (split_dir / "test.uttlist").write_text(
        "\n".join(test_ids.tolist()), encoding="utf-8"
    )

    split_to_ids = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
    }

    def writer_map_from_form_ids(fids):
        writers = sorted({forms_map[fid] for fid in fids if fid in forms_map})
        return {str(w): i for i, w in enumerate(writers)}

    dict_targets = [Path("."), split_dir]
    for split_name, fids in split_to_ids.items():
        payload = json.dumps(writer_map_from_form_ids(fids), ensure_ascii=False)
        for target_dir in dict_targets:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"writers_dict_{split_name}.json").write_text(
                payload, encoding="utf-8"
            )

    print(f"Rows in metafile: {len(rows)}")
    print(f"Line images processed: {len(rows) - missing_images}")
    print(f"Word chunks created: {total_chunks}")
    print(f"Missing images: {missing_images}")
    print(f"Forms total: {len(all_form_ids)}")
    print(f"Train/Val/Test forms: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Wrote: {forms_path}")
    print(f"Wrote: {words_path}")
    print(f"Wrote split files in: {split_dir}")
    print(
        "Wrote writer maps for train/val/test to: "
        f"{Path('.').resolve()} and {split_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
