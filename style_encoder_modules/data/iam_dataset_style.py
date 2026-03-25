import numpy as np
from PIL import Image
import json
import os

from .word_line_dataset import WordLineDataset
from .image_utils import image_resize_PIL


class IAMDataset_style(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = "IAM"
        self.trainset_file = os.path.join(self.basefolder, "IAM", "set_split", "trainset.txt")
        self.valset_file = os.path.join(self.basefolder, "IAM", "set_split", "validationset1.txt")
        self.testset_file = os.path.join(self.basefolder, "IAM", "set_split", "testset.txt")
        self.line_file = os.path.join(self.basefolder, "ascii", "lines.txt")
        self.word_file = "./iam_data/ascii/words.txt"
        self.word_path = os.path.join(self.basefolder, "words")
        self.line_path = os.path.join(self.basefolder, "lines")
        self.forms = "./iam_data/ascii/forms.txt"
        self.split_dir = "./utils/aachen_iam_split"
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, split_name="train", level="word"):
            def _normalize_writer_id(writer_id):
                writer_id = str(writer_id).strip()
                no_leading_zeros = writer_id.lstrip("0") or "0"
                if no_leading_zeros == writer_id:
                    return [writer_id]
                return [writer_id, no_leading_zeros]

            def _to_int_if_possible(value):
                value_str = str(value).strip()
                if value_str.isdigit():
                    return int(value_str)
                return value

            if subset == "train":
                valid_set = np.loadtxt(
                    os.path.join(self.split_dir, "train_val.uttlist"), dtype=str
                )
            elif subset == "val":
                valid_set = np.loadtxt(
                    os.path.join(self.split_dir, "validation.uttlist"), dtype=str
                )
            elif subset == "test":
                valid_set = np.loadtxt(
                    os.path.join(self.split_dir, "test.uttlist"), dtype=str
                )
            else:
                raise ValueError
            if level == "word":
                gtfile = self.word_file
                root_path = self.word_path
                print("root_path", root_path)
                forms = self.forms
            elif level == "line":
                gtfile = self.line_file
                root_path = self.line_path
            else:
                raise ValueError
            gt = []
            form_writer_dict = {}

            dict_candidates = [
                f"./writers_dict_{subset}.json",
                os.path.join(self.basefolder, f"writers_dict_{subset}.json"),
            ]
            if subset == "val":
                dict_candidates.extend([
                    "./writers_dict_test.json",
                    "./writers_dict_train.json",
                    os.path.join(self.basefolder, "writers_dict_test.json"),
                    os.path.join(self.basefolder, "writers_dict_train.json"),
                ])
            dict_path = next(
                (p for p in dict_candidates if os.path.exists(p)),
                None,
            )
            if dict_path is None:
                raise FileNotFoundError(
                    "Could not find any writer dictionary file. Tried: "
                    + ", ".join(dict_candidates)
                )
            with open(dict_path, "r") as f:
                wr_dict = json.load(f)
            reverse_wr_dict = {
                str(v).strip(): _to_int_if_possible(k) for k, v in wr_dict.items()
            }
            missing_writers = set()
            for l in open(forms):
                if not l.startswith("#"):
                    info = l.strip().split()
                    # print('info', info)
                    form_name = info[0]
                    writer_name = info[1]
                    form_writer_dict[form_name] = writer_name
                    # print('form_writer_dict', form_writer_dict)
                    # print('form_name', form_name)
                    # print('writer', writer_name)

            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    name_parts = name.split("-")
                    pathlist = [root_path] + [
                        "-".join(name_parts[: i + 1]) for i in range(len(name_parts))
                    ]
                    # print('name', name)
                    # form =
                    # writer_name = name_parts[1]
                    # print('writer_name', writer_name)

                    if level == "word":
                        line_name = pathlist[-2]
                        del pathlist[-2]

                        if info[1] != "ok":
                            continue

                    elif level == "line":
                        line_name = pathlist[-1]
                    form_name = "-".join(line_name.split("-")[:-1])
                    # print('form_name', form_name)
                    # if (info[1] != 'ok') or (form_name not in valid_set):
                    if form_name not in valid_set:
                        # print(line_name)
                        continue
                    img_path = "/".join(pathlist)

                    transcr = " ".join(info[8:])
                    writer_name = str(form_writer_dict[form_name]).strip()
                    writer_label = None

                    for writer_candidate in _normalize_writer_id(writer_name):
                        if writer_candidate in wr_dict:
                            writer_label = wr_dict[writer_candidate]
                            break
                        if writer_candidate in reverse_wr_dict:
                            writer_label = reverse_wr_dict[writer_candidate]
                            break

                    if writer_label is None:
                        missing_writers.add(writer_name)
                        continue

                    writer_name = _to_int_if_possible(writer_label)

                    gt.append((img_path, transcr, writer_name))
            if missing_writers:
                print(
                    f"Warning: skipped {len(missing_writers)} writer ids "
                    f"missing from {os.path.basename(dict_path)}"
                )
            return gt

        info = gather_iam_info(self, subset, segmentation_level)
        data = []
        widths = []
        for i, (img_path, transcr, writer_name) in enumerate(info):
            if i % 1000 == 0:
                print(
                    "imgs: [{}/{} ({:.0f}%)]".format(
                        i, len(info), 100.0 * i / len(info)
                    )
                )
            #

            try:
                # print('img_path', img_path + '.png')
                img = Image.open(img_path + ".png").convert("RGB")  # .convert('L')
                # print('img shape PIL', img.size)
                # img = image_resize_PIL(img, height=64)

                if img.height < 64 and img.width < 256:
                    img = img
                else:
                    img = image_resize_PIL(img, height=img.height // 2)

                # widths.append(img.size[0])

            except:
                continue

            # except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case
            for cc in special_cases:
                transcr = transcr.replace("|'" + cc, "'" + cc)
                transcr = transcr.replace("|'" + cc.upper(), "'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr, writer_name, img_path)]

        return data
