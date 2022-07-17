import torch
import torch.nn as nn
from gazetracker.models.base import eye_model, landmark_model, regression_head


class GazeTracker(nn.Module):
    """
    Model for combining output from the 2 base models (for eye and landmarks):
    2 fully connected layers (with 8 and 4 hidden units respectively) + 1 final regression head (linear, with 2 units
    for outputting x and y location of gaze on the phone screen).
    """

    def __init__(self):
        super(GazeTracker, self).__init__()

        self.eye_model = eye_model()
        self.lm_model = landmark_model()
        self.regressor_head = regression_head()

    def forward(self, leftEye, rightEye, lms):
        l_eye_feat = torch.flatten(self.eye_model(leftEye), 1)
        r_eye_feat = torch.flatten(self.eye_model(rightEye), 1)
        lm_feat = self.lm_model(lms)

        print(len(l_eye_feat))  # TODO: remove this
        print(l_eye_feat.shape)  # TODO: remove this

        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.regressor_head(combined_feat)
        return out
