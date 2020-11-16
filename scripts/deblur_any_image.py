import numpy as np
from PIL import Image
import click
import os
import cv2

from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = (img - 127.5) / 127.5
    return img

def split_and_predict(g, image):

  (n, m, _) = image.shape

  res = np.zeros(image.shape, dtype=np.uint8)

  for i in range(int(np.ceil(n/256))):
    for j in range(int(np.ceil(m/256))):

      l1 = i * 256
      r1 = np.min([l1+255, image.shape[0] - 1])

      l2 = j * 256
      r2 = np.min([l2+255, image.shape[1] - 1])

      temp = image[l1:r1+1, l2:r2+1]
      temp = preprocess_image(temp)

      tempRes = g.predict(x=np.array([temp]))[0]

      tempRes = cv2.resize(tempRes, (r2-l2+1, r1-l1+1))
      tempRes = deprocess_image(tempRes)

      res[l1:r1+1, l2:r2+1, :] = tempRes

  return res


def deblur(weight_path, input_dir, output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    for image_name in os.listdir(input_dir):
        image = np.array(load_image(os.path.join(input_dir, image_name)))

        generated = split_and_predict(g, image)

        x = image[:, :, :]
        img = generated[:, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save(os.path.join(output_dir, image_name))


@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_dir', help='Image to deblur')
@click.option('--output_dir', help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):
    return deblur(weight_path, input_dir, output_dir)


if __name__ == "__main__":
    deblur_command()
