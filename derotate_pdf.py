import argparse
import os
from pathlib import Path

import cv2
from fpdf import FPDF, Align

from derotate import ADAPTIVE_WINSZ, derotate
from extract_images import add_borders_to_aspect_ratio, extract_pdf_images


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf-file", type=str, required=True)
    parser.add_argument("--img-output-dir", type=str, default="./images")
    parser.add_argument(
        "--max-line-angle",
        type=int,
        default=30,
        help="Maximum allowed angle for approximately horizontal lines",
    )
    parser.add_argument(
        "--adaptive-winsz",
        type=int,
        default=ADAPTIVE_WINSZ,
        help="Window size for adaptive threshold in reduced px",
    )
    parser.add_argument(
        "--min-relative-width",
        type=float,
        default=0.2,
        help="Minimum width relative to maximum width for a line to be considered",
    )
    parser.add_argument(
        "--page-format",
        type=str,
        choices=("A4", "Letter"),
        default="A4",
        help="Page format of output PDF file.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    img_files = extract_pdf_images(args.pdf_file, args.img_output_dir)

    pdf = FPDF(format=args.page_format)
    pdf.set_margins(0, 0, 0)
    for i, img_file in enumerate(img_files):
        print(f"-> derotating image {i+1}/{len(img_files)}")
        img_file_path = os.path.join(args.img_output_dir, img_file)
        derotated_img = derotate(
            img_file_path, args.max_line_angle, args.min_relative_width, args.debug
        )
        derotated_img = add_borders_to_aspect_ratio(derotated_img, pdf.epw / pdf.eph)
        out_img_file = "{0}_{2}{1}".format(*os.path.splitext(img_file) + ("derotate",))
        out_img_file_path = os.path.join(args.img_output_dir, out_img_file)
        cv2.imwrite(out_img_file_path, derotated_img)
        pdf.add_page()
        pdf.image(out_img_file_path, Align.C, None, pdf.epw, pdf.eph)

    pdf_path = f"{Path(args.pdf_file).stem}_derotated.pdf"
    pdf.output(pdf_path)
    print(f"saved derotated PDF to {pdf_path}")


if __name__ == "__main__":
    main()
