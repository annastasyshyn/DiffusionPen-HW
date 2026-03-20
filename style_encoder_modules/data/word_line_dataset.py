import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import isfile
from skimage.transform import resize
import os
from tqdm import tqdm
import random

from .image_utils import image_resize_PIL, centered_PIL


class WordLineDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)
    """
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    """

    def __init__(
        self,
        basefolder: str = "datasets/",  # Root folder
        subset: str = "all",  # Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = "line",  # Type of data to load ('line' or 'word')
        fixed_size: tuple = (128, None),  # Resize inputs to this size
        transforms: list = None,  # List of augmentation transform functions to be applied on each input
        character_classes: list = None,  # If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
    ):

        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None  # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", )

    def __finalize__(self):
        """
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        """
        assert self.setname is not None
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(","))
            self.stopwords = self.stopwords[0]

        save_path = "./IAM_dataset_PIL_style"
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        save_file = "{}/{}_{}_{}.pt".format(
            save_path, self.subset, self.segmentation_level, self.setname
        )  # dataset_path + '/' + set + '_' + level + '_IAM.pt'
        print("save_file", save_file)
        # if isfile(save_file) is False:
        #    data = self.main_loader(self.subset, self.segmentation_level)
        #    torch.save(data, save_file)   #Uncomment this in 'release' version
        # else:
        #    data = torch.load(save_file)

        data = self.main_loader(self.subset, self.segmentation_level)
        self.data = data
        # print('data', self.data)
        self.initial_writer_ids = [d[2] for d in data]

        writer_ids, writer_indices = np.unique(
            [d[2] for d in data], return_inverse=True
        )

        self.data = [
            (img, transcr, int(writer_idx), img_path)
            for (img, transcr, _, img_path), writer_idx in zip(data, writer_indices)
        ]

        self.writer_ids = writer_ids
        self.writer_id_to_index = {wid: idx for idx, wid in enumerate(writer_ids)}

        self.wclasses = len(writer_ids)
        print("Number of writers", self.wclasses)
        if self.character_classes is None:
            res = set()
            # compute character classes given input transcriptions
            for _, transcr, _, _ in tqdm(self.data):
                # print('legth transcr = ', len(transcr))
                res.update(list(transcr))
                self.max_transcr_len = max(self.max_transcr_len, len(transcr))
                # print('self.max_transcr_len', self.max_transcr_len)

            res = sorted(list(res))
            res.append(" ")
            print(
                "Character classes: {} ({} different characters)".format(res, len(res))
            )
            print("Max transcription length: {}".format(self.max_transcr_len))
            self.character_classes = res
            self.max_transcr_len = self.max_transcr_len
        # END FINALIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]

        transcr = self.data[index][1]

        wid = self.data[index][2]

        img_path = self.data[index][3]
        # pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data if p[2] == wid and len(p[1]) > 3]
        negative_samples = [n for n in self.data if n[2] != wid and len(n[1]) > 3]

        positive = random.choice(positive_samples)[0]

        # Make sure you have at least 5 matching images
        if len(positive_samples) >= 5:
            # Randomly select 5 indices from the matching_indices
            random_samples = random.sample(positive_samples, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples]
        else:
            # Handle the case where there are fewer than 5 matching images (if needed)
            # print("Not enough matching images with writer ID", wid)
            positive_samples_ = [p for p in self.data if p[2] == wid]
            # print('len positive samples', len(positive_samples_), 'wid', wid)
            random_samples_ = random.sample(positive_samples_, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples_]

        # pick another image from a different writer
        negative = random.choice(negative_samples)[0]

        img_pos = positive  # image_resize_PIL(positive, height=positive.height // 2)
        img_neg = negative  # image_resize_PIL(negative, height=negative.height // 2)

        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        # print('fheight', fheight, 'fwidth', fwidth)
        if self.subset == "train":
            nwidth = int(np.random.uniform(0.75, 1.25) * img.width)
            nheight = int(
                (np.random.uniform(0.9, 1.1) * img.height / img.width) * nwidth
            )

            nwidth_pos = int(np.random.uniform(0.75, 1.25) * img_pos.width)
            nheight_pos = int(
                (np.random.uniform(0.9, 1.1) * img_pos.height / img_pos.width)
                * nwidth_pos
            )

            nwidth_neg = int(np.random.uniform(0.75, 1.25) * img_neg.width)
            nheight_neg = int(
                (np.random.uniform(0.9, 1.1) * img_neg.height / img_neg.width)
                * nwidth_neg
            )

        else:
            nheight, nwidth = img.height, img.width
            nheight_pos, nwidth_pos = img_pos.height, img_pos.width
            nheight_neg, nwidth_neg = img_neg.height, img_neg.width

        nheight, nwidth = max(4, min(fheight - 16, nheight)), max(
            8, min(fwidth - 32, nwidth)
        )
        nheight_pos, nwidth_pos = max(4, min(fheight - 16, nheight_pos)), max(
            8, min(fwidth - 32, nwidth_pos)
        )
        nheight_neg, nwidth_neg = max(4, min(fheight - 16, nheight_neg)), max(
            8, min(fwidth - 32, nwidth_neg)
        )

        img = image_resize_PIL(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered_PIL(img, (fheight, fwidth), border_value=255.0)

        pixel_values_img = img  # self.processor(img, return_tensors="pt").pixel_values
        pixel_values_img = pixel_values_img  # .squeeze(0)

        img_pos = image_resize_PIL(
            img_pos, height=int(1.0 * nheight_pos), width=int(1.0 * nwidth_pos)
        )
        img_pos = centered_PIL(img_pos, (fheight, fwidth), border_value=255.0)

        img_neg = image_resize_PIL(
            img_neg, height=int(1.0 * nheight_neg), width=int(1.0 * nwidth_neg)
        )
        img_neg = centered_PIL(img_neg, (fheight, fwidth), border_value=255.0)

        pixel_values_pos = (
            img_pos  # self.processor(img_pos, return_tensors="pt").pixel_values
        )
        pixel_values_neg = (
            img_neg  # self.processor(img_neg, return_tensors="pt").pixel_values
        )
        pixel_values_pos = pixel_values_pos  # .squeeze(0)

        pixel_values_neg = pixel_values_neg  # .squeeze(0)

        st_imgs = []
        for s_im in style_images:
            # s_im = image_resize_PIL(s_im, height=s_im.height // 2)
            if self.subset == "train":
                nwidth = int(np.random.uniform(0.75, 1.25) * s_im.width)
                nheight = int(
                    (np.random.uniform(0.9, 1.1) * s_im.height / s_im.width) * nwidth
                )

            else:
                nheight, nwidth = s_im.height, s_im.width

            nheight, nwidth = max(4, min(fheight - 16, nheight)), max(
                8, min(fwidth - 32, nwidth)
            )
            # Load the image and transform it
            s_img = image_resize_PIL(
                s_im, height=int(1.0 * nheight), width=int(1.0 * nwidth)
            )
            s_img = centered_PIL(s_img, (fheight, fwidth), border_value=255.0)
            if self.transforms is not None:
                s_img_tensor = self.transforms(s_img)
            else:
                s_img_tensor = s_img

            st_imgs += [s_img_tensor]

        s_imgs = torch.stack(st_imgs)

        if self.transforms is not None:

            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)

        char_tokens = [self.character_classes.index(c) for c in transcr]
        # print('char_tokens before', char_tokens)
        pad_token = 79

        # padding_length = self.max_transcr_len - len(char_tokens)
        padding_length = 95 - len(char_tokens)
        char_tokens.extend([pad_token] * padding_length)

        # char_tokens += [pad_token] * (self.max_transcr_len - len(char_tokens))
        char_tokens = torch.tensor(char_tokens, dtype=torch.long)

        cla = self.character_classes
        # print('character classes', cla)
        # wid = self.wr_dict[index]
        # print('wid after', index, wid)
        # print('pixel_values_pos', pixel_values_pos.shape)
        # img = outImg
        # save_image(img, 'check_augm.png')
        return (
            img,
            transcr,
            char_tokens,
            wid,
            img_pos,
            img_neg,
            cla,
            s_imgs,
            img_path,
            img,
            img_pos,
            img_neg,
        )  # pixel_values_img, pixel_values_pos, pixel_values_neg

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        (
            img,
            transcr,
            char_tokens,
            wid,
            positive,
            negative,
            cla,
            s_imgs,
            img_path,
            pixel_values_img,
            pixel_values_pos,
            pixel_values_neg,
        ) = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        # transcr_batch = torch.stack(transcr)
        char_tokens_batch = torch.stack(char_tokens)

        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)

        s_imgs = torch.stack(s_imgs)

        pixel_values_img = torch.stack(pixel_values_img)

        pixel_values_pos = torch.stack(pixel_values_pos)
        pixel_values_neg = torch.stack(pixel_values_neg)

        return (
            img,
            transcr,
            char_tokens_batch,
            wid,
            images_pos,
            images_neg,
            cla,
            s_imgs,
            img_path,
            pixel_values_img,
            pixel_values_pos,
            pixel_values_neg,
        )

    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        """
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not

        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        """
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError("The requested fixed image size is invalid!")
            new_img = resize(
                image=img, output_shape=fixed_image_size[::-1], mode="constant"
            )
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print("OUCH")
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode="constant")
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img

    def print_random_sample(self, image, transcription, id, as_saved_files=True):
        import random  #   Create method that will show example images using graphics-in-console (e.g. TerminalImageViewer)
        from PIL import Image

        # Run this with a very low probability
        x = random.randint(0, 10000)
        if x > 5:
            return

        def show_image(img):
            def get_ansi_color_code(r, g, b):
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return (
                    16
                    + (36 * round(r / 255 * 5))
                    + (6 * round(g / 255 * 5))
                    + round(b / 255 * 5)
                )

            def get_color(r, g, b):
                return "\x1b[48;5;{}m \x1b[0m".format(int(get_ansi_color_code(r, g, b)))

            h = 12
            w = int((img.width / img.height) * h)
            img = img.resize((w, h))
            img_arr = np.asarray(img)
            h, w = img_arr.shape  # ,c
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    print(get_color(pix, pix, pix), sep="", end="")
                    # print(get_color(pix[0], pix[1], pix[2]), sep='', end='')
                print()

        if as_saved_files:
            Image.fromarray(np.uint8(image * 255.0)).save(
                "/tmp/a{}_{}.png".format(id, transcription)
            )
        else:
            print('Id = {}, Transcription = "{}"'.format(id, transcription))
            show_image(Image.fromarray(255.0 * image))
            print()
