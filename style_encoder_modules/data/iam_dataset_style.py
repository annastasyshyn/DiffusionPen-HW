import numpy as np
from PIL import Image
import json

from .word_line_dataset import WordLineDataset
from .image_utils import image_resize_PIL


class IAMDataset_style(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = "IAM"
        self.trainset_file = "{}/{}/set_split/trainset.txt".format(
            self.basefolder, self.setname
        )
        self.valset_file = "{}/{}/set_split/validationset1.txt".format(
            self.basefolder, self.setname
        )
        self.testset_file = "{}/{}/set_split/testset.txt".format(
            self.basefolder, self.setname
        )
        self.line_file = "{}/ascii/lines.txt".format(self.basefolder, self.setname)
        self.word_file = "./iam_data/ascii/words.txt".format(
            self.basefolder, self.setname
        )
        self.word_path = "{}/words".format(self.basefolder, self.setname)
        self.line_path = "{}/lines".format(self.basefolder, self.setname)
        self.forms = "./iam_data/ascii/forms.txt"
        # self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
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
                # valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/train_val.uttlist", dtype=str
                )
                # print(valid_set)
            elif subset == "val":
                # valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/validation.uttlist", dtype=str
                )
            elif subset == "test":
                # valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt(
                    "./utils/aachen_iam_split/test.uttlist", dtype=str
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

            dict_path = f"./writers_dict_{subset}.json"
            # open dict file
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
                    f"missing from writers_dict_{subset}.json"
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
