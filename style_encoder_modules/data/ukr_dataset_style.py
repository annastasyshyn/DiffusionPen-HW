import numpy as np
from PIL import Image
import os

from .word_line_dataset import WordLineDataset
from .image_utils import image_resize_PIL


TARGET_HEIGHT = 60
CHUNK_WIDTH = 192
MIN_CHUNK_WIDTH = 64
BLANK_RATIO = 0.95


def _trim_whitespace(img, bg_thresh=240):
    arr = np.array(img.convert("L"))
    col_means = arr.mean(axis=0)
    rightmost = len(col_means) - 1
    while rightmost > 0 and col_means[rightmost] >= bg_thresh:
        rightmost -= 1
    crop_w = max(MIN_CHUNK_WIDTH, rightmost + 1)
    if crop_w < img.width:
        return img.crop((0, 0, crop_w, img.height))
    return img


def _is_mostly_blank(img, bg_thresh=240):
    arr = np.array(img.convert("L"))
    return (arr >= bg_thresh).mean() >= BLANK_RATIO


def _slice_line(img):
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


class UkrDataset_style(WordLineDataset):


    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        transforms,
        split_seed=42,
        val_fraction=0.2,
        lines_subdir="lines/lines",
    ):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = "UKR"
        self.metafile = os.path.join(basefolder, "METAFILE.tsv")
        self.lines_dir = os.path.join(basefolder, lines_subdir)
        self.split_seed = split_seed
        self.val_fraction = val_fraction

        self.num_writers = 0
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        rows = self._read_metafile()

        forms_map = {}
        for fn, _ in rows:
            stem = os.path.splitext(fn)[0]
            parts = stem.split("-")
            if len(parts) >= 2:
                form_id = f"{parts[0]}-{parts[1]}"
                forms_map.setdefault(form_id, parts[1])

        all_forms = sorted(forms_map.keys())
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_forms))
        n_val = int(len(all_forms) * self.val_fraction)
        n_train = len(all_forms) - n_val

        if subset == "train":
            keep = set(all_forms[i] for i in perm[:n_train])
        elif subset == "val":
            keep = set(all_forms[i] for i in perm[n_train:])
        else:
            keep = set(all_forms)

        train_forms = set(all_forms[i] for i in perm[:n_train])
        train_writers = sorted({forms_map[f] for f in train_forms})
        writer_to_label = {w: i for i, w in enumerate(train_writers)}
        self.num_writers = len(train_writers)

        data = []
        n_blank = 0
        n_missing = 0
        n_total_lines = 0

        for fn, transcription in rows:
            stem = os.path.splitext(fn)[0]
            parts = stem.split("-")
            if len(parts) < 4:
                continue

            form_id = f"{parts[0]}-{parts[1]}"
            if form_id not in keep:
                continue

            writer_id = parts[1]
            if writer_id not in writer_to_label:
                continue
            label = writer_to_label[writer_id]

            img_path = os.path.join(self.lines_dir, fn)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                n_missing += 1
                continue

            n_total_lines += 1

            w, h = img.size
            if h > TARGET_HEIGHT:
                new_w = max(1, int(w * TARGET_HEIGHT / h))
                img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

            img = _trim_whitespace(img)
            chunks = _slice_line(img)

            if not transcription:
                transcription = "sample"

            for chunk in chunks:
                if _is_mostly_blank(chunk):
                    n_blank += 1
                    continue
                data.append((chunk, transcription, label, img_path))

            if n_total_lines % 5000 == 0:
                print(
                    f"lines: [{n_total_lines} processed, "
                    f"{len(data)} chunks so far]"
                )

        print(f"UKR loader: {n_total_lines} lines -> {len(data)} chunks "
              f"({n_blank} blank skipped, {n_missing} missing)")
        print(f"Writers in split: {self.num_writers}")
        return data

    def _read_metafile(self):
        rows = []
        with open(self.metafile, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    rows.append((parts[0].strip(), parts[1].strip()))
        return rows
