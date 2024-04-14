# %%

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# %%

img = cv.imread("./rotated.png")
# img = cv.imread("./warped.jpg")
# img = cv.imread("./really_bad.png")
# %%

plt.imshow(img)

# %%

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY_INV)
plt.imshow(img)
# %%

img_norm = img / img.max()
ones = img.sum(axis=1)
# %%

plt.plot(ones)

# %%


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


img_rot = rotate_image(img_norm, -1.2)
plt.imshow(img_rot)
# %%
ones = img_rot.sum(axis=1)
plt.plot(ones)
# %%


def std_objective(angle, image):
    angle = angle[0]
    image = rotate_image(image, angle)
    ones = image.sum(axis=1)
    return -np.std(ones)


a = minimize(std_objective, [0], (img_norm,), "Powell")
# %%
a
# %%

img_rot = rotate_image(img_norm, a.x[0])
plt.imshow(img_rot)
# %%
ones = img_rot.sum(axis=1)
plt.plot(ones)
