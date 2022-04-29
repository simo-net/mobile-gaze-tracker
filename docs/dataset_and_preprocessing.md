## Raw Dataset info

GazeCapture is the first large-scale dataset for eye tracking containing almost 2.5M frames from:
  - more than 1450 people
  - 15 different devices (8 i-phones and 7 i-pads)
The dataset can be downloaded [here](http://gazecapture.csail.mit.edu).

------------

## Preprocessing methods

### MIT split
    We have different participants in train/val/test sets. This ensures that the trained model is
    truly robust and can generalize well.
    Also, all frames that make it to the final dataset contains only valid face detection along
    with valid eye detections (all other frames are discarded). About the filtering conditions, we
    can choose between 2 possibilities:
    1) Less restrictive (all orientations) - 3 filtering conditions:
         - only phone data
         - valid face detections
         - valid eye detections
       Doing so we end up with 1,272,185 total frames from 1247 participants:
         - 1,076,797 train frames from 1081 participants
         - 51,592 validation frames from 45 participants
         - 143,796 test frames from 121 participants
    2) More restrictive (only portraits) - 4 filtering conditions (same as above + 1 more):
         - only phone data
         - valid face detections
         - valid eye detections
         - only portrait orientation
       Doing so we end up with 501,735 total frames from 1241 participants:
         - 427,092 train frames from 1075 participants
         - 19,102 validation frames from 45 participants
         - 55,541 test frames from 121 participants

### Google split
    Frames from each participant are present in all train/val/test sets, but frames related to a
    particular ground truth point do not appear in more than one set.
    Doing so we end up with 501,735 total frames from 1241 participants:
      - 366,940 train frames from 1241 participants
      - 50,946 validation frames from 1219 participants
      - 83,849 test frames from 1233 participants

------------

## Key-points generation

Since the Google model requires eye landmark key points that are not included in the original dataset, 
in order to convert GazeCapture to a dataset usable for this project requires to generate such key points 
for all the files (despite pre-processing the dataset with one of the 2 methods described above).
