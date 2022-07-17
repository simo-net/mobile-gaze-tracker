import os
import joblib
from sklearn.svm import SVR


def train_personalized_regressor(gaze, gaze_estimate,
                                 storage_file: str = 'personal_regressor.joblib'):
    if os.path.splitext(storage_file)[-1] != '.joblib':
        storage_file += '.joblib'
    os.makedirs(os.path.split(storage_file)[0], exist_ok=True)
    if os.path.isfile(storage_file):
        print(f'\nA personalized regressor model was already stored in {storage_file}\n'
              f'The file will be over-written in 10 seconds, interrupt the program NOW if this is not what you want.')
        import time
        time.sleep(10)
    svr = SVR(kernel='rbf', C=20., gamma=0.06)
    svr.fit(gaze_estimate, gaze)
    joblib.dump(svr, storage_file)


def test_personalized_regressor(gaze_estimate,
                                storage_file: str = 'personal_regressor.joblib'):
    if not os.path.isfile(storage_file) or os.path.splitext(storage_file)[-1] != '.joblib':
        raise Exception(f'The file {storage_file} does not exist or has wrong extension (must be .joblib).')
    svr = joblib.load(storage_file)
    gaze_refined = svr.predict(gaze_estimate)
    return gaze_refined
