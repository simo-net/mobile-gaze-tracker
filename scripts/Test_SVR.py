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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

root = os.environ['SLURM_TMPDIR']

# ## Load model and weights
f = root + '/svr13_gt_fin/test/images/'
weight_file = '../Checkpoints/GoogleCheckpoint_MITSplit.ckpt'
print(weight_file)
print(f, len(glob(f + '*.jpg')))

# In[7]:
model = gazetrack_model()
w = torch.load(weight_file)['state_dict']
model.cuda()
model.load_state_dict(w)
model.eval()

# In[8]:
preds, gt = [], []
ctr = 1
model.eval()
test_dataset = Gaze_Capture(f, split='test')
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
all_files = glob(f + "*.jpg")
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

# In[10]:
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
np.mean(mean_errs)

# In[11]:
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

# ## SVR

# Add hook to [5] for pre ReLU and [6] for after ReLU
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.combined_model[6].register_forward_hook(get_activation('out'))


def train_svr(file):
    file = file.replace("test", "train")
    dataset = Gaze_Capture(file, split="test", verbose=False)
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=10,
        pin_memory=False,
        shuffle=False,
    )

    preds, gt, dot_nums = [], [], []
    calib_preds, calib_gt = [], []
    for j in loader:
        leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()
        with torch.no_grad():
            pred = list(model.forward(leye, reye, kps).cpu().detach().numpy())
        pred = list(activation["out"].detach().cpu().numpy())
        preds = preds + pred
        gt.extend(target.cpu().detach().numpy())

    gt = np.array(gt)
    preds = np.array(preds)
    reg = MultiOutputRegressor(SVR(kernel="rbf", C=20, gamma=0.06))

    reg.fit(preds, gt)

    return reg


def comp_pred_test_svr(fname, ct=False):
    reg = train_svr(fname)

    f = fname.replace("train", "test")
    if ct:
        f = fname

    test_dataset = Gaze_Capture(f, split="test", verbose=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=10,
        pin_memory=False,
        shuffle=False,
    )

    preds_pre, preds_final, gt, dot_nums = [], [], [], []
    cent_fin = []

    calib_preds, calib_gt = [], []

    for j in test_dataloader:
        leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()

        with torch.no_grad():
            pred = model.forward(leye, reye, kps)
        pred = pred.cpu().detach().numpy()
        act = list(activation["out"].cpu().detach().numpy())

        pred_fin = reg.predict(act)

        preds_final.extend(pred_fin)
        preds_pre.extend(pred)
        gt.extend(target.cpu().detach().numpy())

    preds_pre = np.array(preds_pre)
    preds_final = np.array(preds_final)

    pts = np.unique(gt, axis=0)

    c = get_colors(len(pts))
    random.shuffle(c)

    gt = np.array(gt)
    dist_pre = euc(preds_pre, gt)
    dist_final = euc(preds_final, gt)

    out = [dist_pre, dist_final, gt, preds_pre, preds_final, pts, c]

    return out


# In[18]:
svr_out = {}
for i in tqdm(files[:]):
    svr_out[i] = comp_pred_test_svr(i)

# Overall
means_pre = []
means_post = []
for idx, i in enumerate(svr_out):
    means_pre.extend(svr_out[i][0])
    means_post.extend(svr_out[i][1])
print("Mean without SVR: ", np.mean(means_pre), " Mean after SVR: ", np.mean(means_post))

# In[25]:
mean_errs_pre = []
mean_errs_final = []
for i in svr_out:
    mean_errs_pre.append(np.mean(svr_out[i][0]))
    mean_errs_final.append(np.mean(svr_out[i][1]))

plt.figure(figsize=(15, 10))
plt.title('SVR Mean Comparison (13 Point Calibration)')
ctr = 0
plt.hlines(y=np.mean(mean_errs_pre), xmin=0, xmax=len(mean_errs_pre), color='b', linestyles='dashed',
           label="Overall Base Model Mean Error: " + str(np.round(np.mean(mean_errs_pre), 3)) + " cm")
plt.hlines(y=np.median(mean_errs_final), xmin=0, xmax=len(mean_errs_pre), color='k', linestyles='dashed',
           label="Overall Post SVR Mean Error: " + str(np.round(np.mean(mean_errs_final), 3)) + " cm")
for i in range(len(means_pre)):
    if means_post[i] <= means_pre[i]:
        plt.vlines(x=ctr, ymin=means_post[i], ymax=means_pre[i], colors='green')
    else:
        plt.vlines(x=ctr, ymin=means_pre[i], ymax=means_post[i], colors='red')
    ctr += 1
plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_pre, s=15, label="Base Model Mean Error", color='b')
plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_final, s=15, label="Post SVR Mean Error", color='black')
plt.xlabel('Subject id')
plt.ylabel('Mean Euclidean Distance (cm)')
plt.ylim(0)
plt.legend()
plt.show()

# ## Few outputs
mt = 0
for idx in tqdm(list(svr_out.keys())[:10]):
    i = idx
    plot_comp(svr_out[i][2], svr_out[i][3], svr_out[i][4], svr_out[i][5], svr_out[i][6],
              ['Mean Euc error pre svr: ' + str(np.mean(svr_out[i][0])),
               'Mean Euc error after svr: ' + str(np.mean(svr_out[i][1]))], cent='none', lines=True)

# # Random tests
idx = np.random.randint(len(files))
meta = json.load(open(glob(files[idx].replace('images', 'meta') + "*.json")[0]))
print("device: ", meta['device'])
print("id: ", idx)
print("Num files: ", len(glob(files[idx] + "*.jpg")))
print("Sample:")
plt.imshow(plt.imread(glob(files[idx] + "*.jpg")[0]))

# In[34]:
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
