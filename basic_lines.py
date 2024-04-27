# %%
from datetime import datetime as dt

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


def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0 / (max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height], dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    return (rval + 0.5).astype(int) if as_integer else rval


# %%

region_points = []
span_points = []
for y_avg in smoothed_y_values:
    valid_indices = np.where(y_avg > 0)[0]
    points = np.array([(index, y_avg[index]) for index in valid_indices])
    sampled_points = points[::50]
    region_points.append(sampled_points)
    sampled_points = pix2norm(image.shape, sampled_points)
    sampled_points = sampled_points.reshape(-1, 1, 2)
    span_points.append(sampled_points)

# %%
point_image = rotated_image.copy()

for region in region_points:
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

PAGE_MARGIN_Y = 50
PAGE_MARGIN_X = 50


def calculate_page_extents(image):
    height, width = image.shape[:2]
    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax, ymax = (width - xmin), (height - ymin)
    pagemask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(pagemask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)
    page_outline = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    return pagemask, page_outline


def keypoints_from_samples(small, pagemask, page_outline, span_points):
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0
    for points in span_points:
        _, evec = cv2.PCACompute(points.reshape((-1, 2)), mean=None, maxComponents=1)
        weight = np.linalg.norm(points[-1] - points[0])
        all_evecs += evec * weight
        all_weights += weight
    evec = all_evecs / all_weights
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])
    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2))).reshape(
        (-1, 2)
    )
    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)
    px0, px1 = px_coords.min(), px_coords.max()
    py0, py1 = py_coords.min(), py_coords.max()
    # [px0,px1,px1,px0] for first bit of p00,p10,p11,p01
    x_dir_coeffs = np.pad([px0, px1], 2, mode="symmetric")[2:].reshape(-1, 1)
    # [py0,py0,py1,py1] for second bit of p00,p10,p11,p01
    y_dir_coeffs = np.repeat([py0, py1], 2).reshape(-1, 1)
    corners = np.expand_dims((x_dir_coeffs * x_dir) + (y_dir_coeffs * y_dir), 1)
    xcoords, ycoords = [], []
    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords, py_coords = np.dot(pts, np.transpose([x_dir, y_dir])).T
        xcoords.append(px_coords - px0)
        ycoords.append(py_coords.mean() - py0)
    return corners, np.array(ycoords), xcoords


pagemask, page_outline = calculate_page_extents(image)
corners, ycoords, xcoords = keypoints_from_samples(
    image, pagemask, page_outline, span_points
)
# %%

FOCAL_LENGTH = 1.8


def K():
    return np.array(
        [
            [FOCAL_LENGTH, 0, 0],
            [0, FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


# TODO add span points to params
def get_default_params(corners, ycoords, xcoords):
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
    span_counts = [*map(len, xcoords)]
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array(cubic_slopes).flatten(),
            ycoords.flatten(),
        )
        + tuple(xcoords)
    )
    return (page_width, page_height), span_counts, params


# %%

rough_dims, span_counts, params = get_default_params(corners, ycoords, xcoords)

dstpoints = np.vstack(
    (corners[0].reshape(1, 1, 2),)
    + tuple(point.reshape(-1, 1, 2) for point in span_points)
)

# %%

RVEC_IDX = slice(0, 3)  # Index of rvec in params vector (slice: pair of values)
TVEC_IDX = slice(3, 6)  # Index of tvec in params vector (slice: pair of values)
CUBIC_IDX = slice(
    6, 8
)  # Index of cubic slopes in params vector (slice: pair of values)


def make_keypoint_index(span_counts):
    nspans, npts = len(span_counts), sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start : start + end, 1] = 8 + i
        start = end
    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
    return keypoint_index


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


def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0
    return project_xy(xy_coords, pvec)


def optimise_params(small, dstpoints, span_counts, params, debug_lvl):
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts) ** 2)

    print("  initial objective is", objective(params))
    print("  optimizing", len(params), "parameters...")
    start = dt.now()
    res = minimize(objective, params, method="Powell")
    end = dt.now()
    print(f"  optimization took {round((end - start).total_seconds(), 2)} sec.")
    print(f"  final objective is {res.fun}")
    params = res.x
    return params


# %%

OUTPUT_ZOOM = 1.0
REMAP_DECIMATE = 16


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    return i + factor - rem if rem else i


# TODO continue here
def remap(img, page_dims, params):
    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)
    width = round_nearest_multiple(height * page_dims[0] / page_dims[1], REMAP_DECIMATE)
    print("  output will be {}x{}".format(width, height))
    height_small, width_small = np.floor_divide([height, width], REMAP_DECIMATE)
    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)
    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    page_xy_coords = np.hstack(
        (
            page_x_coords.flatten().reshape((-1, 1)),
            page_y_coords.flatten().reshape((-1, 1)),
        )
    )
    page_xy_coords = page_xy_coords.astype(np.float32)
    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)
    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)
    image_x_coords = cv2.resize(
        image_x_coords, (width, height), interpolation=cv2.INTER_CUBIC
    )
    image_y_coords = cv2.resize(
        image_y_coords, (width, height), interpolation=cv2.INTER_CUBIC
    )
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    remapped = cv2.remap(
        img_gray,
        image_x_coords,
        image_y_coords,
        cv2.INTER_CUBIC,
        None,
        cv2.BORDER_REPLICATE,
    )
    return remapped


# %%

_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

baseline_img = remap(binary_image, rough_dims, params)
# %%

plt.imshow(baseline_img)
# %%

optimal_params = optimise_params(binary_image, dstpoints, span_counts, params, 1)

# %%


def get_page_dims(corners, rough_dims, params):
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten()) ** 2)

    res = minimize(objective, dims, method="Powell")
    dims = res.x
    print("  got page dims", dims[0], "x", dims[1])
    return dims


# %%


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


page_dims = get_page_dims(corners, rough_dims, optimal_params)
print(page_dims)
if np.any(page_dims < 0):
    # Fallback: see https://github.com/lmmx/page-dewarp/issues/9
    print("Got a negative page dimension! Falling back to rough estimate")
    page_dims = rough_dims

img_remapped = remap(binary_image, rough_dims, optimal_params)
# img_remapped = minmax(img_remapped)

plt.imshow(img_remapped)
plt.show()
plt.imshow(baseline_img)
# %%
