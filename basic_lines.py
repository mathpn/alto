# %%
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np

# Read the image
# image = cv2.imread("rotated.png")
# image = cv2.imread("warped.jpg")
# image = cv2.imread("okayish.png")
image = cv2.imread("./really_bad_cropped.png")
output_image = image.copy()


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Perform line detection using Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=10)

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

# %%

image = rotated_image  # XXX

line_thickness = int(image.shape[0] / 150)
binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
for line in filtered_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(binary_image, (x1, y1), (x2, y2), 255, line_thickness)

horizontal_projection = np.sum(binary_image, axis=1)
threshold_value = 0.5 * np.max(horizontal_projection)
peaks = np.where(horizontal_projection > threshold_value)[0]

# Define regions based on identified peaks and their neighboring areas
regions = []
current_region = []
for i in range(len(peaks) - 1):
    if peaks[i + 1] - peaks[i] > 1:
        if current_region:
            regions.append((current_region[0], current_region[-1]))
            current_region = []
        continue
    current_region.extend(range(peaks[i], peaks[i + 1]))

if current_region:
    regions.append((current_region[0], current_region[-1]))

plt.imshow(binary_image)
plt.show()

plt.plot(horizontal_projection)
plt.show()
# Group the lines within each region
binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grouped_lines = []
for region in regions:
    region_lines = []
    region_x_min = image.shape[1]
    region_x_max = 0
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        y_min, y_max = sorted((y1, y2))
        if y_min < region[0] or y_max > region[1]:
            continue
        x_min, x_max = sorted((x1, x2))
        region_x_min = min(region_x_min, x_min)
        region_x_max = max(region_x_max, x_max)
        region_lines.append(line)

    if (region_x_max - region_x_min) / image.shape[1] < 0.5:  # XXX
        continue

    for line in region_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary_image, (x1, y1), (x2, y2), 255, line_thickness)

    grouped_lines.append(region_lines)

plt.imshow(binary_image)

# %%
# Calculate the average y-coordinate for each x-coordinate along the horizontal span
image_height = image.shape[0]
image_width = image.shape[1]

y_values = np.arange(image_width)
y_values = []
for region_lines in grouped_lines:
    y_sum = np.zeros(image_width)
    count = np.zeros(image_width)
    for line in region_lines:
        x1, y1, x2, y2 = line[0]
        y_sum[min(x1, x2) : max(x1, x2) + 1] += np.linspace(y1, y2, abs(x2 - x1) + 1)
        count[min(x1, x2) : max(x1, x2) + 1] += 1
    avg_y = np.divide(y_sum, count, out=np.zeros_like(y_sum), where=count != 0)
    y_values.append(avg_y)

plt.plot(np.hstack(y_values))

# %%

# Smooth the average y-values using a Savitzky-Golay filter
smoothed_y_values = []
for y_avg in y_values:
    smoothed_y = np.zeros_like(y_avg)
    smoothed_y[y_avg > 0] = savgol_filter(y_avg[y_avg > 0], 250, 1)  # XXX parameter
    smoothed_y_values.append(smoothed_y)

plt.plot(np.hstack(smoothed_y_values))

# %%

region_points = []
sampled_region_points = []
for y_avg in smoothed_y_values:
    valid_indices = np.where(y_avg > 0)[0]
    points = np.array([(index, y_avg[index]) for index in valid_indices])
    sampled_points = points[::50]
    region_points.append(points)
    sampled_region_points.append(sampled_points)

# %%
point_image = rotated_image.copy()

for region in sampled_region_points:
    for point in region:
        cv2.circle(
            point_image,
            (int(point[0]), int(point[1])),
            radius=20,
            color=(0, 255, 255),
            thickness=-1,
        )

plt.figure(figsize=(10, 10))
plt.imshow(point_image)
# %%

pca_points, true_points = [], []
for points_array in sampled_region_points:
    true_points.append(points_array)  # XXX wasteful
    # Compute the PCA with one component
    mean, eigenvectors = cv2.PCACompute(points_array, mean=None, maxComponents=1)

    direction_vector = eigenvectors[0]
    unit_direction = direction_vector / np.linalg.norm(direction_vector)

    inv_direction = np.array([-unit_direction[1], unit_direction[0]])
    projected_points = np.outer(
        np.dot(points_array, unit_direction), unit_direction
    ) + np.outer(np.dot(mean, inv_direction), inv_direction)
    pca_points.append(projected_points)
    plt.scatter(points_array[:, 0], points_array[:, 1])
    plt.scatter(projected_points[:, 0], projected_points[:, 1])
    plt.show()

pca_points = np.vstack(pca_points)
true_points = np.vstack(true_points)
# %%

FOCAL_LENGTH = 1.2


def K():
    return np.array(
        [
            [FOCAL_LENGTH, 0, 0],
            [0, FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def get_default_params(corners):
    page_width, page_height = [np.linalg.norm(corners[i] - corners[0]) for i in (1, -1)]
    cubic_slopes = [0.0, 0.0]  # initial guess for the cubic has no slope
    # object points of flat page in 3D coordinates
    corners_object3d = np.array(
        [
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0],
        ]
    )
    # estimate rotation and translation from four 2D-to-3D point correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d, corners, K(), np.zeros(5))
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array(cubic_slopes).flatten(),
        )
    )
    return (page_width, page_height), params


# %%
img_height, img_width, *_ = image.shape
corners = np.array(
    [
        [[0, 0.0]],
        [[img_width, 0]],
        [[img_width, img_height]],
        [[0, img_height]],
    ]
)
# corners = np.array(
#     [
#         [[0, 0.0]],
#         [[1, 0]],
#         [[1, 1]],
#         [[0, 1]],
#     ]
# )

(width, height), params = get_default_params(corners)

# %%

RVEC_IDX = slice(0, 3)  # Index of rvec in params vector (slice: pair of values)
TVEC_IDX = slice(3, 6)  # Index of tvec in params vector (slice: pair of values)
CUBIC_IDX = slice(
    6, 8
)  # Index of cubic slopes in params vector (slice: pair of values)


def project_xy(xy_coords, pvec):
    """
    Get cubic polynomial coefficients given by:

      f(0) = 0, f'(0) = alpha
      f(1) = 0, f'(1) = beta
    """
    alpha, beta = tuple(pvec[CUBIC_IDX])
    poly = np.array([alpha + beta, -2 * alpha - beta, alpha, 0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    image_points, _ = cv2.projectPoints(
        objpoints,
        pvec[RVEC_IDX],
        pvec[TVEC_IDX],
        K(),
        np.zeros(5),
    )
    return image_points


# %%


def remap(img, params):
    height, width = img.shape
    page_x_range = np.linspace(0, width, 100)  # XXX
    page_y_range = np.linspace(0, height, 100)
    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    page_xy_coords = np.hstack(
        (
            page_x_coords.flatten().reshape((-1, 1)),
            page_y_coords.flatten().reshape((-1, 1)),
        )
    )
    page_xy_coords = page_xy_coords.astype(np.float32)
    image_points = project_xy(page_xy_coords, params)
    # image_points = norm2pix(img.shape, image_points, False)
    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)
    image_x_coords = cv2.resize(
        image_x_coords, (width, height), interpolation=cv2.INTER_CUBIC
    )
    image_y_coords = cv2.resize(
        image_y_coords, (width, height), interpolation=cv2.INTER_CUBIC
    )
    remapped = cv2.remap(
        img,
        image_x_coords,
        image_y_coords,
        cv2.INTER_CUBIC,
        None,
        cv2.BORDER_REPLICATE,
    )
    return remapped


# %%

img_remapped = remap(binary_image, params)
plt.imshow(img_remapped)
# %%

plt.imshow(img_remapped)
# %%

(width, height), params = get_default_params(corners)

baseline_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, baseline_img = cv2.threshold(baseline_img, 100, 255, cv2.THRESH_BINARY)
plt.imshow(baseline_img)
plt.show()

# %%

ref_points = project_xy(true_points, params)
pca_points = pca_points.copy()


def objective(pvec, *args):
    ppts = project_xy(pca_points, pvec)
    loss = np.mean((ppts - ref_points) ** 2)
    return loss


print(f"baseline = {objective(params):.4f}")
print("--------------------------------")
# %%
a = minimize(objective, params, "Powell")
print(a)
