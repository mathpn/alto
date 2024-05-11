import argparse
import os

import cv2

from dewarp import Config, dewarp
from extract_images import extract_pdf_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--img-output-dir", type=str, default="./images")
    args = parser.parse_args()

    config = Config()

    img_files = extract_pdf_images(args.pdf_file, args.img_output_dir)
    print(img_files)
    for i, img_file in enumerate(img_files):
        dewarped_img = dewarp(img_file, config, args.debug)
        out_img_file = os.path.join(args.img_output_dir, "dewarped_img%05i.png" % i)
        cv2.imwrite(out_img_file, dewarped_img)


if __name__ == "__main__":
    main()
