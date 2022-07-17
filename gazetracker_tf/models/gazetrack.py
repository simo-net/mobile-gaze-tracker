import tensorflow as tf
from gazetracker_tf.models.base import eye_model, landmark_model, regression_head


class GazeTracker(tf.keras.Model):

    def __init__(self):
        super(GazeTracker, self).__init__()

        self.eye_model = eye_model()
        self.lm_model = landmark_model()
        self.regressor_head = regression_head()

    def call(self, batch):
        leftEye, rightEye, lms = batch
        l_eye_feat = self.eye_model(leftEye)
        r_eye_feat = self.eye_model(rightEye)
        lm_feat = self.lm_model(lms)

        print(len(l_eye_feat))  # TODO: remove this
        print(l_eye_feat.shape)  # TODO: remove this

        combined_feat = tf.concat([l_eye_feat, r_eye_feat, lm_feat], axis=1)
        out = self.regressor_head(combined_feat)
        return out


# class GazeTracker(tf.keras.Model):
#
#     def __init__(self):
#         super(GazeTracker, self).__init__()
#
#         # ------------------------- Eye Model -------------------------
#         # CONV 1
#         self.conv1 = tf.keras.layers.Conv2D(32, input_shape=(128,128,3), kernel_size=(7,7), strides=(2,2), activation='relu')  # TODO: check input shape
#         self.norm1 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         self.pool1 = tf.keras.layers.AvgPool2D(pool_size=(2,2))
#         self.drop1 = tf.keras.layers.Dropout(0.02)
#         # CONV 2
#         self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=(2,2), activation='relu')
#         self.norm2 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         self.pool2 = tf.keras.layers.AvgPool2D(pool_size=(2,2))
#         self.drop2 = tf.keras.layers.Dropout(0.02)
#         # CONV 3
#         self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu')
#         self.norm3 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         self.pool3 = tf.keras.layers.AvgPool2D(pool_size=(2,2))
#         self.drop3 = tf.keras.layers.Dropout(0.02)
#         self.flat = tf.keras.layers.Flatten()
#
#         # ----------------------- Landmark model ----------------------
#         # FC 1
#         self.fc1 = tf.keras.layers.Dense(128, input_shape=(8,), activation='relu')  # TODO: check input shape
#         self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         # FC 2
#         self.fc2 = tf.keras.layers.Dense(16, activation='relu')
#         self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         # FC 3
#         self.fc3 = tf.keras.layers.Dense(16, activation='relu')
#         self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.9)
#
#         # ---------------------- Regression Head ----------------------
#         # FC 4
#         self.fc4 = tf.keras.layers.Dense(8, input_shape=(128+128+16,), activation='relu')  # TODO: check input shape
#         self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         self.dr4 = tf.keras.layers.Dropout(0.12)
#         # FC 5
#         self.fc5 = tf.keras.layers.Dense(4, activation='relu')
#         self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.9)
#         # FC 6
#         self.fc6 = tf.keras.layers.Dense(2, activation=None)
#
#     def call(self, inputs):
#         leftEye, rightEye, lms = inputs
#
#         # ************ Left Eye ************
#         l_eye_feat = self.conv1(leftEye)
#         l_eye_feat = self.norm1(l_eye_feat)
#         l_eye_feat = self.pool1(l_eye_feat)
#         l_eye_feat = self.drop1(l_eye_feat)
#         l_eye_feat = self.conv2(l_eye_feat)
#         l_eye_feat = self.norm2(l_eye_feat)
#         l_eye_feat = self.pool2(l_eye_feat)
#         l_eye_feat = self.drop2(l_eye_feat)
#         l_eye_feat = self.conv3(l_eye_feat)
#         l_eye_feat = self.norm3(l_eye_feat)
#         l_eye_feat = self.pool3(l_eye_feat)
#         l_eye_feat = self.drop3(l_eye_feat)
#
#         print(len(l_eye_feat))  # TODO: remove this
#         print(l_eye_feat.shape)  # TODO: remove this
#
#         # ************ Right Eye ***********
#         r_eye_feat = self.conv1(rightEye)
#         r_eye_feat = self.norm1(r_eye_feat)
#         r_eye_feat = self.pool1(r_eye_feat)
#         r_eye_feat = self.drop1(r_eye_feat)
#         r_eye_feat = self.conv2(r_eye_feat)
#         r_eye_feat = self.norm2(r_eye_feat)
#         r_eye_feat = self.pool2(r_eye_feat)
#         r_eye_feat = self.drop2(r_eye_feat)
#         r_eye_feat = self.conv3(r_eye_feat)
#         r_eye_feat = self.norm3(r_eye_feat)
#         r_eye_feat = self.pool3(r_eye_feat)
#         r_eye_feat = self.drop3(r_eye_feat)
#
#         # ************ Landmarks ***********
#         lm_feat = self.fc1(lms)
#         lm_feat = self.bn1(lm_feat)
#         lm_feat = self.fc2(lm_feat)
#         lm_feat = self.bn2(lm_feat)
#         lm_feat = self.fc3(lm_feat)
#         lm_feat = self.bn3(lm_feat)
#
#         combined_feat = tf.concat([l_eye_feat, r_eye_feat, lm_feat], axis=1)
#         out = self.fc4(combined_feat)
#         out = self.bn4(out)
#         out = self.dr4(out)
#         out = self.fc5(out)
#         out = self.bn5(out)
#         out = self.fc6(out)
#         return out
