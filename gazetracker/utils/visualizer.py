import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import colorsys


def make_img(fname):
    img1 = plt.imread(glob(fname + "*.jpg")[np.random.randint(len(glob(fname + "*.jpg")))])
    img2 = plt.imread(glob(fname + "*.jpg")[np.random.randint(len(glob(fname + "*.jpg")))])
    img3 = plt.imread(glob(fname + "*.jpg")[np.random.randint(len(glob(fname + "*.jpg")))])
    img4 = plt.imread(glob(fname + "*.jpg")[np.random.randint(len(glob(fname + "*.jpg")))])
    col = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2, img1.shape[2]))
    print(col.shape)
    col[0:img1.shape[0], 0:img1.shape[1], :] = img1
    col[0:img1.shape[0], img1.shape[1]:img1.shape[1] * 2, :] = img3

    col[img1.shape[0]:img1.shape[0] * 2, 0:img1.shape[1], :] = img2
    col[img1.shape[0]:img1.shape[0] * 2, img1.shape[1]:img1.shape[1] * 2, :] = img4
    return col / 255.


def euc(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=1))


def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_pts(fname, gt, preds, pts, c, title, cent=False):
    cols = np.zeros((len(gt), 3))
    cents = np.zeros_like(preds)

    for i in range(len(pts)):
        cols[np.where(np.all(gt == pts[i], axis=1))] = c[i]
        if cent:
            cents[np.where(np.all(gt == pts[i], axis=1))] = np.mean(preds[np.where(np.all(gt == pts[i], axis=1))],
                                                                    axis=0)

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(make_img(fname))
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.grid(color='gray', linestyle='dashed')
    plt.scatter(0, 0, marker='*', s=200, label="Camera")
    plt.scatter(preds[:, 0], preds[:, 1], c=cols, s=10, label="Model predictions")
    plt.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200, label="Ground truth locations")
    if cent:
        plt.scatter(cents[:, 0], cents[:, 1], c=cols, marker='1', s=100, label="Predictions centroid")
    plt.xlabel('distance from the camera at origin in x direction (cm)')
    plt.ylabel('distance from the camera at origin in y direction (cm)')
    plt.axis('scaled')
    plt.xlim(-2.5, 4)
    plt.ylim(-13, 0.2)
    plt.legend()
    plt.show()


def plot_comp(gt, preds1, preds2, pts, c, title=['', ''], cent='none', lines=False):
    cols = np.zeros((len(gt), 3))
    cents1 = np.zeros_like(preds1)
    cents2 = np.zeros_like(preds2)
    for i in range(len(pts)):
        cols[np.where(np.all(gt == pts[i], axis=1))] = c[i]
        if cent:
            cents1[np.where(np.all(gt == pts[i], axis=1))] = np.mean(preds1[np.where(np.all(gt == pts[i], axis=1))],
                                                                     axis=0)
            cents2[np.where(np.all(gt == pts[i], axis=1))] = np.mean(preds2[np.where(np.all(gt == pts[i], axis=1))],
                                                                     axis=0)

    f, axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(22, 11))
    f.patch.set_facecolor('white')
    ax1, ax2, ax3 = axes
    ax1.set_title(title[0])
    ax1.grid(color='gray', linestyle='dashed')
    ax1.scatter(0, 0, marker='*', s=200, label='Camera')
    if cent == 'none' or cent == 'both':
        ax1.scatter(preds1[:, 0], preds1[:, 1], c=cols, s=10, label="Base Model Predictions")
    ax1.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200, label="Ground Truths")
    if cent == 'both' or cent == 'only':
        ax1.scatter(cents1[:, 0], cents1[:, 1], c=cols, marker='1', s=70, label="Base Model Prediction Centroids")
    ax1.set_xlabel('distance from the camera at origin in x direction (cm)')
    ax1.set_ylabel('distance from the camera at origin in y direction (cm)')
    ax1.set_xlim([-3, 5])
    ax1.set_ylim([-12, 0.5])
    ax1.set_aspect('equal')

    ax2.set_title(title[1])
    ax2.grid(color='gray', linestyle='dashed')
    ax2.scatter(0, 0, marker='*', s=200, label='Camera')
    if cent == 'none' or cent == 'both':
        ax2.scatter(preds2[:, 0], preds2[:, 1], c=cols, s=10, label="SVR Predictions")
    ax2.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200, label="Ground Truths")
    if cent == 'both' or cent == 'only':
        ax2.scatter(cents2[:, 0], cents2[:, 1], c=cols, marker='1', s=70, label="SVR Prediction Centroids")
    ax2.set_xlabel('distance from the camera at origin in x direction (cm)')
    ax2.set_ylabel('distance from the camera at origin in y direction (cm)')
    ax2.set_xlim([-3, 5])
    ax2.set_ylim([-12, 0.5])
    ax2.set_aspect('equal')

    ############## LINES #########

    if lines:
        ax3.set_title("Movement of points")
        ax3.grid(color='gray', linestyle='dashed')
        ax3.scatter(0, 0, marker='*', s=200, label='Camera')
        if cent == 'none' or cent == 'both':
            ax3.scatter(preds1[:, 0], preds1[:, 1], c=cols, s=10, label="Base Model Predictions")
            ax3.scatter(preds2[:, 0], preds2[:, 1], c=cols, marker="p", s=30, label="SVR Predictions")
        if cent == 'both' or cent == 'only':
            ax3.scatter(cents1[:, 0], cents1[:, 1], c=cols, marker='1', s=70, label="Base Model Prediction Centroids")
            ax3.scatter(cents2[:, 0], cents2[:, 1], c=cols, marker='p', s=70, label="SVR Prediction Centoids")

        for i in range(len(preds1)):
            if cent == 'none' or cent == 'both':
                ax3.plot([preds1[i][0], preds2[i][0]], [preds1[i][1], preds2[i][1]], c=cols[i], linewidth=0.5)
            if cent == 'only' or cent == 'both':
                ax3.plot([cents1[i][0], cents2[i][0]], [cents1[i][1], cents2[i][1]], c=cols[i], linewidth=0.5)

        ax3.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200)
        ax3.set_xlabel('distance from the camera at origin in x direction (cm)')
        ax3.set_ylabel('distance from the camera at origin in y direction (cm)')
        ax3.axis('scaled')
        ax3.set_xlim([-3, 5])
        ax3.set_ylim([-12, 0.5])
        ax3.set_aspect('equal')
        ax3.legend(loc=3)
        ax2.legend(loc=3)
        ax1.legend(loc=3)
        plt.show()
