import tensorflow as tf


def eye_model():
    """
    Model for computing single eye image:
    ConvNet tower consisting of 3 convolutional layers
    (with 7×7, 5×5 and 3×3 kernel sizes,
    strides of 2, 2 and 1,
    and 32, 64, and 128 output channels, respectively).
    ReLUs were used as non-linearities.
    """
    # CONV 1
    conv1 = tf.keras.layers.Conv2D(32, input_shape=(128, 128, 3), kernel_size=(7,7), strides=(2,2), activation='relu')  # TODO: check input shape
    conv1 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv1)
    conv1 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(conv1)
    conv1 = tf.keras.layers.Dropout(0.02)(conv1)
    # CONV 2
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=(2,2), activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2)
    conv2 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(conv2)
    conv2 = tf.keras.layers.Dropout(0.02)(conv2)
    # CONV 3
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu')(conv2)
    conv3 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv3)
    conv3 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(conv3)
    conv3 = tf.keras.layers.Dropout(0.02)(conv3)
    out = tf.keras.layers.Flatten()(conv3)
    return out


def landmark_model():
    """
    Model for the eye-corner landmarks:
    3 successive fully connected layers (with 128, 16 and 16 hidden units respectively).
    """
    # FC 1
    fc1 = tf.keras.layers.Dense(128, input_shape=(8,), activation='relu')  # TODO: check input shape
    fc1 = tf.keras.layers.BatchNormalization(momentum=0.9)(fc1)
    # FC 2
    fc2 = tf.keras.layers.Dense(16, activation='relu')(fc1)
    fc2 = tf.keras.layers.BatchNormalization(momentum=0.9)(fc2)
    # FC 3
    fc3 = tf.keras.layers.Dense(16, activation='relu')(fc2)
    fc3 = tf.keras.layers.BatchNormalization(momentum=0.9)(fc3)
    return fc3


def regression_head():
    """
    Model for combining output from the 2 base models (for eye and landmarks):
    2 fully connected layers (with 8 and 4 hidden units respectively) + 1 final regression head (linear, with 2 units
    for outputting x and y location of gaze on the phone screen).
    """
    # FC 4
    fc4 = tf.keras.layers.Dense(8, input_shape=(128+128+16,), activation='relu')  # TODO: check input shape
    fc4 = tf.keras.layers.BatchNormalization(momentum=0.9)(fc4)
    fc4 = tf.keras.layers.Dropout(0.12)(fc4)
    # FC 5
    fc5 = tf.keras.layers.Dense(4, activation='relu')(fc4)
    fc5 = tf.keras.layers.BatchNormalization(momentum=0.9)(fc5)
    # FC 6
    fc6 = tf.keras.layers.Dense(2, activation=None)(fc5)
    return fc6
