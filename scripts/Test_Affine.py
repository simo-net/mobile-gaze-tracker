import os
import json
import torch
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from gazetracker.models.gazetrack import gazetrack_model
from gazetracker.dataset.loader import Gaze_Capture
from gazetracker.utils.visualizer import euc, get_colors, plot_pts, plot_comp

import cv2

root = os.environ['SLURM_TMPDIR']

# ## Load model and weights
file_root = root + '/svr13_gt_fin/test/'
weight_file = '../Checkpoints/GoogleCheckpoint_MITSplit.ckpt'
print(weight_file)
print(file_root, len(glob(file_root + 'images/*.jpg')))

# In[5]:
model = gazetrack_model()
w = torch.load(weight_file)['state_dict']
model.cuda()
model.load_state_dict(w)
model.eval()

# In[6]:
preds, gt = [], []
ctr = 1
model.eval()
test_dataset = Gaze_Capture(file_root, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False, )
for j in tqdm(test_dataloader):
    leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

    with torch.no_grad():
        pred = model.forward(leye, reye, kps)
    pred = pred.cpu().detach().numpy()
    preds.extend(pred)

    gt.extend(target.cpu().detach().numpy())

preds = np.array(preds)

gt = np.array(gt)
dist = euc(preds, gt)
print("Mean Euclidean Distance: ", dist.mean())

# ## Total Test
all_files = glob(file_root + "images/*.jpg")
all_files = [i[:-10] for i in all_files]
files = np.unique(all_files)
print('Found ', len(all_files), ' images from ', len(files), ' subjects.')

fnames = []
nums = []
for i in tqdm(files):
    fnames.append(i)
    nums.append(len(glob(i + "*.jpg")))
fnames = np.array(fnames)
nums = np.array(nums)
ids = np.argsort(nums)
ids = ids[::-1]
fnames_sorted = fnames[ids]
nums_sorted = nums[ids]
files = fnames_sorted.copy()
nums_sorted[0], nums_sorted[-1], sum(nums_sorted)

# In[8]:
total_test = {}
ep = 0

for idx in tqdm(range(len(files))):
    preds, gt = [], []
    ctr = 1
    f = files[idx]
    test_dataset = Gaze_Capture(f, split='test', verbose=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, num_workers=10, pin_memory=False, shuffle=False, )

    for j in test_dataloader:
        leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

        with torch.no_grad():
            pred = model.forward(leye, reye, kps)

        pred = pred.cpu().detach().numpy()
        preds.extend(pred)

        gt.extend(target.cpu().detach().numpy())

    preds = np.array(preds)
    pts = np.unique(gt, axis=0)

    gt = np.array(gt)
    dist = euc(preds, gt)
    total_test[str(idx) + "_" + str(ep)] = [dist, gt, preds, pts]

mean_errs = []
for i in total_test:
    mean_errs.append(np.mean(total_test[i][0]))
print(np.mean(mean_errs))

# In[9]:
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title(
    weight_file + "\nMean Euc dist: " + str(np.mean(mean_errs)) + "\nMedian Euc dist: " + str(np.median(mean_errs)))
plt.scatter([i for i in range(len(mean_errs))], mean_errs, s=10)
plt.hlines(y=np.mean(mean_errs), xmin=0, xmax=len(mean_errs), color='r')
plt.hlines(y=np.median(mean_errs), xmin=0, xmax=len(mean_errs), color='y')
plt.xlabel('Subject id')
plt.ylabel('Mean Euclidean Distance')
plt.ylim(0)

plt.subplot(1, 2, 2)
plt.title(files[0][:files[0].rfind('/')])
plt.scatter([i for i in range(len(mean_errs))], [len(total_test[i][0]) for i in total_test], s=30)
plt.xlabel('Subject id')
plt.ylabel('Number of datapoints')
plt.show()


# ## Affine

def train_affine(file, avg=False):
    file = file.replace('test', 'train')

    dataset = Gaze_Capture(file, split='test', verbose=False)
    loader = DataLoader(dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False, )

    preds, gt = [], []
    for j in loader:
        leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

        with torch.no_grad():
            pred = list(model.forward(leye, reye, kps).cpu().detach().numpy())
        preds = preds + pred

        gt.extend(target.cpu().detach().numpy())

    gt = np.array(gt)
    preds = np.array(preds)
    cent = np.zeros_like(preds)

    calib_preds = []
    calib_gt = []

    if avg:
        for i in np.unique(dot_nums):
            calib_preds.append(np.mean(preds[np.where(dot_nums == i)], axis=0))
            cent[np.where(dot_nums == i)] = np.mean(preds[np.where(dot_nums == i)], axis=0)
            calib_gt.append(np.mean(gt[np.where(dot_nums == i)], axis=0))
        calib_preds = np.array(calib_preds)
        calib_gt = np.array(calib_gt)
    else:
        calib_preds = preds.copy()
        calib_gt = gt.copy()

    cent = np.array(cent)
    trans = cv2.estimateAffine2D(calib_preds, calib_gt, method=cv2.RANSAC)[0]
    return trans, cent


def comp_pred_test(fname, avg=True, ct=False):
    trans, cent = train_affine(fname, avg)

    f = fname.replace('train', 'test')
    if ct:
        f = fname
    test_dataset = Gaze_Capture(f, split='test', verbose=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False, )

    preds_pre, preds_final, gt = [], [], []
    for j in test_dataloader:
        leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

        with torch.no_grad():
            pred = model.forward(leye, reye, kps)

        pred = pred.cpu().detach().numpy()
        preds_pre.extend(pred)
        gt.extend(target.cpu().detach().numpy())

    preds_pre = np.array(preds_pre)

    preds_final = np.dot(trans[:, 0:2], preds_pre.T).T + trans[:, 2]

    preds_pre = np.array(preds_pre)
    preds_final = np.array(preds_final)
    pts = np.unique(gt, axis=0)

    c = get_colors(len(pts))
    random.shuffle(c)

    gt = np.array(gt)
    dist_pre = euc(preds_pre, gt)
    dist_final = euc(preds_final, gt)

    out = [dist_pre, dist_final, gt, preds_pre, preds_final, pts, c, cent]

    return out


# In[11]:
affine_out = {}
for i in tqdm(files[:]):
    affine_out[i] = comp_pred_test(i, avg=False)

# In[14]:

# Overall
means_pre = []
means_post = []
for idx, i in enumerate(affine_out):
    means_pre.extend(affine_out[i][0])
    means_post.extend(affine_out[i][1])
print("Mean without affine: ", np.mean(means_pre), " Mean after affine: ", np.mean(means_post))

# In[25]:
mean_errs_pre = []
mean_errs_final = []
for i in affine_out:
    mean_errs_pre.append(np.mean(affine_out[i][0]))
    mean_errs_final.append(np.mean(affine_out[i][1]))

print(len(mean_errs_pre), len(mean_errs_final))

# In[26]:
mean_errs_pre = []
mean_errs_final = []
for i in affine_out:
    mean_errs_pre.append(np.mean(affine_out[i][0]))
    mean_errs_final.append(np.mean(affine_out[i][1]))

plt.figure(figsize=(15, 10))
plt.title('Affine Mean Comparison (13 Point Calibration)')
ctr = 0
plt.hlines(y=np.mean(mean_errs_pre), xmin=0, xmax=len(mean_errs_pre), color='b', linestyles='dashed',
           label="Overall Base Model Mean Error: " + str(np.round(np.mean(mean_errs_pre), 3)) + " cm")
plt.hlines(y=np.median(mean_errs_final), xmin=0, xmax=len(mean_errs_pre), color='k', linestyles='dashed',
           label="Overall Post Affine Mean Error: " + str(np.round(np.mean(mean_errs_final), 3)) + " cm")
for i in range(len(mean_errs_pre)):
    if mean_errs_final[i] <= mean_errs_pre[i]:
        plt.vlines(x=ctr, ymin=mean_errs_final[i], ymax=mean_errs_pre[i], colors='green')
    else:
        plt.vlines(x=ctr, ymin=mean_errs_pre[i], ymax=mean_errs_final[i], colors='red')
    ctr += 1
plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_pre, s=15, label="Base Model Mean Error", color='b')
plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_final, s=15, label="Post Affine Mean Error",
            color='black')
plt.xlabel('Subject id')
plt.ylabel('Mean Euclidean Distance (cm)')
plt.ylim(0)
plt.legend()
plt.show()

# ## Few outputs
for i in tqdm(list(affine_out.keys())[:10]):
    plot_comp(affine_out[i][2], affine_out[i][3], affine_out[i][4], affine_out[i][5], affine_out[i][6],
              ['Mean Euc error pre affine: ' + str(np.mean(affine_out[i][0])),
               'Mean Euc error after affine: ' + str(np.mean(affine_out[i][1]))], cent='both', lines=True)

# # Random tests
idx = np.random.randint(len(files))
meta = json.load(open(glob(files[idx].replace('images', 'meta') + "*.json")[0]))
print("device: ", meta['device'])
print("id: ", idx)
print("Num files: ", len(glob(files[idx] + "*.jpg")))
print("Sample:")
plt.figure()
plt.imshow(plt.imread(glob(files[idx] + "*.jpg")[0]))
plt.show()

# In[18]:
preds, gt = [], []
ctr = 1
f = files[idx]
fs = glob(f + "*.jpg")
test_dataset = Gaze_Capture(f, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False, )

preds, gt = [], []
for j in test_dataloader:
    leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

    with torch.no_grad():
        pred = model.forward(leye, reye, kps)
    pred = pred.cpu().detach().numpy()
    preds.extend(pred)

    gt.extend(target.cpu().detach().numpy())

preds = np.array(preds)
pts = np.unique(gt, axis=0)

c = get_colors(len(pts))
random.shuffle(c)

gt = np.array(gt)
dist = euc(preds, gt)
print("Mean Euclidean Distance: ", dist.mean())
plot_pts(f, gt, preds, pts, c, "Mean euc: " + str(dist.mean()), cent=True)
