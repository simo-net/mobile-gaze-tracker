import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from gazetracker.dataset.loader import GazeCapture
from gazetracker.models.base import eye_model, landmark_model


class lit_gazetrack_model(pl.LightningModule):
    """
    Model for combining output from the 2 base models (for eye and landmarks):
    2 fully connected layers (with 8 and 4 hidden units respectively) + 1 final regression head (linear, with 2 units
    for outputting x and y location of gaze on the phone screen).
    """

    def __init__(self, data_path, save_path, batch_size, lr, logger, workers=20):
        super(lit_gazetrack_model, self).__init__()

        self.lr = lr
        self.batch_size = batch_size

        self.data_path = data_path
        self.save_path = save_path
        print("Data path: ", data_path)

        self.workers = workers

        PARAMS = {'batch_size': self.batch_size,
                  'init_lr': self.lr,
                  'data_path': self.data_path,
                  'save_path': self.save_path,
                  'scheduler': "Plateau"}
        logger.log_hyperparams(PARAMS)

        self.eye_model = eye_model()
        self.lm_model = landmark_model()
        self.combined_model = nn.Sequential(nn.Linear(512+512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4, 2))

    def forward(self, leftEye, rightEye, lms):
        l_eye_feat = torch.flatten(self.eye_model(leftEye), 1)
        r_eye_feat = torch.flatten(self.eye_model(rightEye), 1)

        lm_feat = self.lm_model(lms)

        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat)
        return out

    def train_dataloader(self):
        train_dataset = GazeCapture(os.path.join(self.data_path, "train"))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)
        self.logger.log_hyperparams({'Num_train_files': len(train_dataset)})
        return train_loader

    def val_dataloader(self):
        val_dataset = GazeCapture(os.path.join(self.data_path, "val"))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)
        self.logger.log_hyperparams({'Num_val_files': len(val_dataset)})
        return val_loader

    def training_step(self, batch, batch_idx):
        l_eye, r_eye, kps, y = batch
        y_hat = self.forward(l_eye, r_eye, kps)

        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.logger.experiment.log_metric('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        l_eye, r_eye, kps, y = batch
        y_hat = self.forward(l_eye, r_eye, kps)

        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        self.logger.experiment.log_metric('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        # scheduler = ExponentialLR(optimizer, gamma=0.64, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
