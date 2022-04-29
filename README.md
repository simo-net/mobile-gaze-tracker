An implementation of Google's paper - "Accelerating eye movement research via accurate and affordable smartphone eye tracking" for GSoC 2021 

# Useful links

### This repository was readapted from:
https://github.com/DSSR2/gaze-track

### All details are available at:
https://dssr2.github.io/gaze-track

### Original paper (nature 2020) can be found at:
https://www.nature.com/articles/s41467-020-18360-5.pdf

### Dataset can be downloaded from:
http://gazecapture.csail.mit.edu


# Data information
The dataset can be downloaded at the [project website](http://gazecapture.csail.mit.edu/download.php). In the dataset, we include data for 1474 unique subjects. Each numbered directory represents a recording session from one of those subjects. Numbers were assigned sequentially, although some numbers are missing for various reasons (e.g., test recordings, duplicate subjects, or incomplete uploads).

Inside each directory is a collection of sequentially-numbered images (in the `frames` subdirectory) and JSON files for different pieces of metadata, described below. Many of the variables in the JSON files are arrays, where each element is associated with the frame numbered the same as the index.

We only make use of frames where the subject's device was able to detect both the user's [face](https://developer.apple.com/reference/avfoundation/avcapturemetadataoutputobjectsdelegate) and [eyes](https://developer.apple.com/reference/coreimage/cidetector) using Apple's built-in libraries. Some subjects had *no* frames with face and eye detections at all. There are 2,445,504 total frames and 1,490,959 with complete Apple detections. For this reason, some frames will be "missing" generated data.

The dataset is split into three pieces, by subject (i.e., recording number): training, validation, and test.

Following is a description of each variable:

### appleFace.json, appleLeftEye.json, appleRightEye.json
These files describe bounding boxes around the detected face and eyes, logged at recording time using Apple libraries. "Left eye" refers to the subject's physical left eye, which appears on the right side of the image.

- `X`, `Y`: Position of the top-left corner of the bounding box (in pixels). In `appleFace.json`, this value is relative to the top-left corner of the full frame; in `appleLeftEye.json` and `appleRightEye.json`, it is relative to the top-left corner of the *face crop*.
- `W`, `H`: Width and height of the bounding box (in pixels).
- `IsValid`: Whether or not there was actually a detection. 1 = detection; 0 = no detection.

### dotInfo.json
- `DotNum`: Sequence number of the dot (starting from 0) being displayed during that frame.
- `XPts`, `YPts`: Position of the center of the dot (in points; see `screen.json` documentation below for more information on this unit) from the top-left corner of the screen.
- `XCam`, `YCam`: Position of the center of the dot in our prediction space. The position is measured in centimeters and is relative to the camera center, assuming the camera remains in a fixed position in space across all device orientations. I.e., `YCam` values will be negative for portrait mode frames (`Orientation` == 1) since the screen is below the camera, but values will be positive in upside-down portrait mode (`Orientation` == 2) since the screen is above the camera. See Section 4.1 and Figure 6 for more information.
- `Time`: Time (in seconds) since the displayed dot first appeared on the screen.

### faceGrid.json
These values describe the "face grid" input features, which were generated from the Apple face detections. Within a 25 x 25 grid of 0 values, these parameters describe where to draw in a box of 1 values to represent the position and size of the face within the frame.

- `X`, `Y`: Position of the top-left corner of the face box (1-indexed, within a 25 x 25 grid).
- `W`, `H`: Width and height of the face box.
- `IsValid`: Whether the data is valid (1) or not (0). This is equivalent to the intersection of the associated `IsValid` arrays in the apple*.json files (since we required samples to have Apple face and eye detections).

### frames.json
The filenames of the frames in the `frames` directory. This information may also be generated from a sequence number counting from 0 to `TotalFrames` - 1 (see `info.json`).

### info.json
- `TotalFrames`: The total number of frames for this subject.
- `NumFaceDetections`: The number of frames in which a face was detected.
- `NumEyeDetections`: The number of frames in which eyes were detected.
- `Dataset`: "train," "val," or "test."
- `DeviceName`: The name of the device used in the recording.

### motion.json
A stream of motion data (accelerometer, gyroscope, and magnetometer) recorded at 60 Hz, only while frames were being recorded. See Apple's [CMDeviceMotion](https://developer.apple.com/reference/coremotion/cmdevicemotion) class for a description of the values. `DotNum` (counting from 0) and `Time` (in seconds, from the beginning of that dot's recording) are recorded as well.

### screen.json
- `H`, `W`: Height and width of the active screen area of the app (in points). This allows us to account for the iOS "Display Zoom" feature (which was used by some subjects) as well as larger status bars (e.g., when a Personal Hotspot is enabled) and split screen views (which was not used by any subjects). See [this](https://developer.apple.com/library/content/documentation/2DDrawing/Conceptual/DrawingPrintingiOS/GraphicsDrawingOverview/GraphicsDrawingOverview.html) and [this](https://www.paintcodeapp.com/news/ultimate-guide-to-iphone-resolutions) page for more information on the unit "points."
- `Orientation`: The orientation of the interface, as described by the enumeration [UIInterfaceOrientation](https://developer.apple.com/reference/uikit/uiinterfaceorientation), where:
  - 1: portrait
  - 2: portrait, upside down (iPad only)
  - 3: landscape, with home button on the right
  - 4: landscape, with home button on the left
