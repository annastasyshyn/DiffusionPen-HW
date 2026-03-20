import numpy as np
from PIL import Image


def image_resize_PIL(img, height=None, width=None):
    if height is None and width is None:
        return img  # No resizing needed

    original_width, original_height = img.size

    if height is not None and width is None:
        scale = height / original_height
        new_width = int(original_width * scale)
        new_height = height
    elif width is not None and height is None:
        scale = width / original_width
        new_width = width
        new_height = int(original_height * scale)
    else:
        new_width = width
        new_height = height

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    # resized_img.save('res.png')
    return resized_img


def centered_PIL(word_img, tsize, centering=(0.5, 0.5), border_value=None):

    height = tsize[0]
    width = tsize[1]
    # print('word_img.size', word_img.size)
    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height - word_img.height
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h - pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h / 2, word_img.height - (diff_h - diff_h / 2)
        padh = (0, 0)
    diff_w = width - word_img.width
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.width - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)

    # print('word_img.size, padw, padh', word_img.size, padw, padh)
    res = Image.new("RGB", (width, height), color=(255, 255, 255))
    # res.save('background.png')

    res.paste(word_img, (padw[0], padh[0]))

    return res
