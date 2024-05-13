import argparse

from extract_images import extract_pdf_images


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf-file", type=str, required=True)
    parser.add_argument("--img-output-dir", type=str, default="./images")
    args = parser.parse_args()

    extract_pdf_images(args.pdf_file, args.img_output_dir)


if __name__ == "__main__":
    main()
