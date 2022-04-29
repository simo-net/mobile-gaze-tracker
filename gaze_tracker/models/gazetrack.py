import torch
import torch.nn as nn
from gaze_tracker.models.base import eye_model, landmark_model


class gazetrack_model(nn.Module):
    """
    Model for combining output from the 2 base models (for eye and landmarks):
    2 fully connected layers (with 8 and 4 hidden units respectively) + 1 final regression head (linear, with 2 units
    for outputting x and y location of gaze on the phone screen).
    """

    def __init__(self):
        super(gazetrack_model, self).__init__()

        self.eye_model = eye_model()
        self.lm_model = landmark_model()
        self.combined_model = nn.Sequential(nn.Linear(512 + 512 + 16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4, 2), )

    def forward(self, leftEye, rightEye, lms):
        l_eye_feat = torch.flatten(self.eye_model(leftEye), 1)
        r_eye_feat = torch.flatten(self.eye_model(rightEye), 1)

        lm_feat = self.lm_model(lms)

        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat)
        return out
