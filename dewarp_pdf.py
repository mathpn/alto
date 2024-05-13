import argparse
import os

import cv2

from dewarp import Config, dewarp
from extract_images import extract_pdf_images


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf-file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--img-output-dir", type=str, default="./images")
    args = parser.parse_args()

    config = Config()

    img_files = extract_pdf_images(args.pdf_file, args.img_output_dir)
    print(img_files)
    for i, img_file in enumerate(img_files):
        print(f"-> dewarping image {i+1}/{len(img_files)}")
        img_file_path = os.path.join(args.img_output_dir, img_file)
        dewarped_img = dewarp(img_file_path, config, args.debug)
        out_img_file = "{0}_{2}{1}".format(*os.path.splitext(img_file) + ("dewarp",))
        out_img_file_path = os.path.join(args.img_output_dir, out_img_file)
        cv2.imwrite(out_img_file_path, dewarped_img)


if __name__ == "__main__":
    main()
