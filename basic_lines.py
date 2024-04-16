# %%
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
import numpy as np

# Read the image
image = cv2.imread("rotated.png")
# image = cv2.imread("warped.jpg")
# image = cv2.imread("./really_bad.png")
output_image = image.copy()


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Perform line detection using Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)

# Filter lines based on their orientation
filtered_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Calculate angle of the line
    if np.abs(angle) < 30:  # Adjust the threshold angle as needed
        filtered_lines.append(line)

# Draw filtered lines
for i, line in enumerate(filtered_lines):
    x1, y1, x2, y2 = line[0]
    cv2.line(
        output_image, (x1, y1), (x2, y2), (0, 255, 0), 2
    )  # Green color, line thickness 2

# %%

plt.figure(figsize=(10, 15))
plt.imshow(output_image)
# %%

# Aggregate all line segments into a single array of points
all_points = np.concatenate([line.reshape(-1, 2) for line in filtered_lines])

# %%


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


angles, sizes = [], []
for line in filtered_lines:
    x1, y1, x2, y2 = line[0]
    size = distance.euclidean([x1, y1], [x2, y2])
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    angles.append(angle)
    sizes.append(size)

average_angle = (np.array(angles) * np.array(sizes)).sum() / np.sum(sizes)
print(np.mean(angles))

# Rotate the image
rows, cols = image.shape[:2]
rotated_image = rotate_image(image, average_angle)

plt.figure(figsize=(10, 15))
combined = np.hstack((image, rotated_image))
plt.imshow(combined)

