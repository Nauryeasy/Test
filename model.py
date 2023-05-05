from PIL import Image
from torchvision import transforms
from utils import *

convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()


def make_sticker(sticker, emoji, target):
    img_tensor = model(sticker_to_tensor(sticker), emoji_to_tensor(emoji), emoji_to_tensor(target))
    img = convert_image(img_tensor)
    img.show()


def sticker_to_tensor(sticker):
    img = Image.open(sticker)
    return convert_tensor(img)


def emoji_to_tensor(emoji):
    return unique_emojies.index(emoji)


if __name__ == '__main__':
    make_sticker('stickers/file_0.webp', '🤔', '🤔')
