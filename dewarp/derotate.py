import argparse

import cv2
import numpy as np
from scipy.spatial import distance

from dewarp import ADAPTIVE_WINSZ, get_horizontal_lines, save_debug_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image,
        rot_mat,
        image.shape[1::-1],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result


def calc_average_angle(image, horizontal_lines, min_relative_width: float, debug: bool):
    min_line_size = min_relative_width * image.shape[1]
    angles, sizes, valid_lines = [], [], []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        size = distance.euclidean([x1, y1], [x2, y2])
        if size < min_line_size:
            continue

        valid_lines.append(line)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        sizes.append(size)

    if debug:
        debug_image = image.copy()
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        for line in valid_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_debug_image(debug_image, "02_lines")  # XXX separate debug from dewarp

    average_angle = (np.array(angles) * np.array(sizes)).sum() / (1e-3 + np.sum(sizes))
    return average_angle


def derotate(input_image: str, max_angle: int, min_relative_width: float, debug: bool):
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    aspect_ratio = height / width
    small_width = 1500  # XXX
    small_height = int(small_width * aspect_ratio)
    gray_small = cv2.resize(gray, (small_width, small_height))

    horizontal_lines = get_horizontal_lines(gray_small, max_angle, debug)
    average_angle = calc_average_angle(
        gray_small, horizontal_lines, min_relative_width, debug
    )
    print(f"average angle: {average_angle:.2f}")
    derotated_img = rotate_image(gray, average_angle)
    return derotated_img


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument(
        "--max-line-angle",
        type=int,
        default=15,
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
        default=0.075,
        help="Minimum width relative to maximum width for a line to be considered",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    derotated_img = derotate(
        args.input_image, args.max_line_angle, args.min_relative_width, args.debug
    )

    derotated_img_binary = cv2.adaptiveThreshold(
        derotated_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        args.adaptive_winsz,
        25,
    )
    output_fpath = "./derotate_output.png"
    cv2.imwrite(output_fpath, derotated_img_binary)
    print(f"saved derotated image to {output_fpath}")


if __name__ == "__main__":
    main()
