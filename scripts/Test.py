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
from gazetracker.utils.visualizer import euc, get_colors

root = os.environ['SLURM_TMPDIR']

# ## Load model and weights
model = gazetrack_model()
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')

weights = torch.load("./Checkpoints/GoogleCheckpoint_1.ckpt", map_location=dev)['state_dict']
model.load_state_dict(weights)
model.to(dev)
model.eval()

# ## Run predictions on entire test set
file_root = root + "/gt_fin/test/"
test_dataset = Gaze_Capture(file_root, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False)

preds, gt = [], []
for j in tqdm(test_dataloader):
    leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)

    with torch.no_grad():
        pred = model.forward(leye, reye, kps)
    pred = pred.cpu().detach().numpy()
    preds.extend(pred)
    gt.extend(target.cpu().detach().numpy())

preds = np.array(preds)
pts = np.unique(gt, axis=0)

gt = np.array(gt)
dist = euc(preds, gt)
print("Mean Euclidean Distance: ", dist.mean())

# In[12]:
all_files = glob(root + "/gt_fin/test/images/*.jpg")
all_files = [i[:-10] for i in all_files]
files = np.unique(all_files)
print('Found ', len(all_files), ' images from ', len(files), ' subjects.')

# In[13]:
idx = np.random.randint(len(files))
meta = json.load(open(glob(files[idx].replace('images', 'meta') + "*.json")[0]))
print("device: ", meta['device'])
print("id: ", idx)
print("Num files: ", len(glob(files[idx] + "*.jpg")))
print("Sample:")
plt.figure()
plt.imshow(plt.imread(glob(files[idx] + "*.jpg")[0]))
plt.show()

# In[15]:
preds, gt = [], []
ctr = 1
f = files[idx]
# f = root+'/dataset/train/images/'
fs = glob(f + "*.jpg")
test_dataset = Gaze_Capture(f, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False, )

for j in tqdm(test_dataloader):
    leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)

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

# In[16]:
cols = np.zeros((len(gt), 3))
for i in range(len(pts)):
    cols[np.where(np.all(gt == pts[i], axis=1))] = c[i]

plt.figure(figsize=(12, 12))
plt.grid(color='gray', linestyle='dashed')
plt.scatter(0, 0, marker='*', s=200)
plt.scatter(preds[:, 0], preds[:, 1], c=cols, s=10)
plt.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200)
plt.xlabel('distance from the camera at origin in x direction (cm)')
plt.ylabel('distance from the camera at origin in y direction (cm)')
plt.axis('scaled')
plt.xlim(-2.5, 4)
plt.ylim(-13, 0.2)
plt.show()

# ## Total Test
total_test = {}
for idx in tqdm(range(len(files))):
    preds, gt = [], []
    ctr = 1
    f = files[idx]
    test_dataset = Gaze_Capture(f, split='test', verbose=False)
    test_dataloader = DataLoader(test_dataset, batch_size=30, num_workers=10, pin_memory=False, shuffle=False, )

    for j in test_dataloader:
        leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)

        with torch.no_grad():
            pred = model.forward(leye, reye, kps)
        pred = pred.cpu().detach().numpy()
        preds.extend(pred)

        gt.extend(target.cpu().detach().numpy())

    preds = np.array(preds)
    pts = np.unique(gt, axis=0)

    gt = np.array(gt)
    dist = euc(preds, gt)
    total_test[idx] = [dist, gt, preds, pts]

# In[19]:
mean_errs = []
for i in total_test:
    mean_errs.append(np.mean(total_test[i][0]))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.scatter([i for i in range(len(mean_errs))], mean_errs, s=10)
plt.hlines(y=np.mean(mean_errs), xmin=0, xmax=len(mean_errs), color='r')
plt.xlabel('Subject id')
plt.ylabel('Mean Euclidean Distance')
plt.subplot(1, 2, 2)
plt.scatter([i for i in range(len(mean_errs))], [len(total_test[i][0]) for i in total_test], s=30)
plt.xlabel('Subject id')
plt.ylabel('Number of datapoints')
plt.show()

print(np.mean(mean_errs))
print(np.std(mean_errs) / np.sqrt(len(mean_errs)))

dists = [total_test[i][0] for i in total_test]
plt.figure(figsize=(24, 12))
plt.boxplot(dists)
plt.xlabel('Subject id')
plt.ylabel('Euc Distance')
plt.show()

# In[ ]:
mean_errs = []
for i in total_test:
    mean_errs.extend(total_test[i][0])
plt.figure(figsize=(24, 12))
plt.scatter([i for i in range(len(mean_errs))], mean_errs, s=10)
plt.hlines(y=np.mean(mean_errs), xmin=0, xmax=len(mean_errs), color='r')
plt.xlabel('Test point id')
plt.ylabel('Mean Euclidean Distance')
plt.show()

print(np.mean(mean_errs))

# In[ ]:
pts = np.unique(gt, axis=0)
cols = np.zeros((len(gt), 3))
for i in range(len(pts)):
    cols[np.where(np.all(gt == pts[i], axis=1))] = c[i]

plt.figure(figsize=(12, 12))
plt.grid(color='gray', linestyle='dashed')
plt.scatter(0, 0, marker='*', s=200)
plt.scatter(preds[:, 0], preds[:, 1], c=cols, s=10)
plt.scatter(gt[:, 0], gt[:, 1], c=cols, marker='+', s=200)
plt.xlabel('distance from the camera at origin in x direction (cm)')
plt.ylabel('distance from the camera at origin in y direction (cm)')
plt.axis('scaled')
plt.xlim(-2.5, 4)
plt.ylim(-13, 0.2)
plt.show()
