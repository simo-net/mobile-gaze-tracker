import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import json, pickle
import tensorflow as tf
from gazetracker_tf.models.gazetrack import GazeTracker
from gazetracker_tf.dataset.loader import GazeCapture, create_dataloader
from gazetracker_tf.utils.runner import NoGPUError, train, test

# Usage:
# python ./train-test-gazetrack-TF.py  -data_dir ~/data/..  -log_dir ./logs/..  -batch_size 20


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training a 2D-CNN with ResNet architecture on two-channeled (ON and OFF) images from the\n'
                    'data_dir and save results to the given log_dir.')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory where all data is stored (must contain "train", "val" and "test" folders).')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory where to store the results of training.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size to use for loading the data on GPU memory. Default is 256.')
    parser.add_argument('--max_num_epochs', type=int, default=30000,
                        help='Number of epochs to use for training the model. Default is 30000.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to use as early-stopping patience. Default is 10.')
    parser.add_argument('-lr_init', type=float, default=0.016,
                        help='Initial learning rate value (then changed by scheduler).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility. Default is None.')
    return parser.parse_args()


def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()

    # Define output log directory ------------------------------------------------------------------#
    os.makedirs(args.log_dir, exist_ok=True)

    # Check if GPU is available --------------------------------------------------------------------#
    if not tf.config.list_physical_devices('GPU'):
        raise NoGPUError('No tensorflow-compatible GPU found. Aborting.')

    # Specify a random seed for reproducibility ----------------------------------------------------#
    if args.seed is not None:
        tf.random.set_seed(args.seed)

    # Load the training and validation data --------------------------------------------------------#
    print('\n\nLoading training and validation data...')
    # Load training dataset ------------------------------------------------------------------------#
    train_dataset = GazeCapture(root=os.path.join(args.data_dir, "train"), size=(128,128), verbose=True)
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = GazeCapture(root=os.path.join(args.data_dir, "val"), size=(128,128), verbose=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True)
    NUM_TRAIN_SAMPLES = len(train_dataset)
    NUM_VAL_SAMPLES = len(val_dataset)
    num_train_samples = NUM_TRAIN_SAMPLES - NUM_VAL_SAMPLES
    num_train_batches = num_train_samples // args.batch_size
    num_val_batches = NUM_VAL_SAMPLES // args.batch_size

    # Build the model ------------------------------------------------------------------------------#
    print('\n\nBuilding the model...')
    model = GazeTracker()
    model.summary()

    # Store the model
    model_file = os.path.join(args.log_dir, 'model.json')
    with open(model_file, 'w+') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    # Train the model ------------------------------------------------------------------------------#
    print('\n\nTraining the model...')
    model, history = train(model=model,
                           training_set=train_loader,
                           validation_set=val_loader,
                           max_num_epochs=args.max_num_epochs,
                           steps_per_epoch=num_train_batches,
                           validation_steps=num_val_batches,
                           patience=args.patience, lr_init=0.016,
                           checkpoint_file=os.path.join(args.log_dir, 'checkpoint'))
    # Store training history
    history_file = os.path.join(args.log_dir, 'history')
    with open(history_file, 'wb') as outfile:
        pickle.dump(history, outfile)  # Read as: history = pickle.load(open(history_file), "rb")
    # Store trained weights
    weights_file = os.path.join(args.log_dir, 'final_weights.hdf5')
    model.save_weights(weights_file)

    # Load the test data ---------------------------------------------------------------------------#
    print('\n\nLoading the test data...')
    del train_dataset, val_dataset, train_loader, val_loader
    test_dataset = GazeCapture(root=os.path.join(args.data_dir, "test"), size=(128,128), verbose=True)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    NUM_TEST_SAMPLES = len(test_dataset)
    num_test_batches = NUM_TEST_SAMPLES // args.batch_size

    # Evaluate model performance -------------------------------------------------------------------#
    print('\n\nEvaluating the model...')
    results = test(model=model,
                   test_set=test_loader,
                   evaluation_steps=num_test_batches)
    # Store the test results
    eval_file = os.path.join(args.log_dir, 'evaluation.json')
    with open(eval_file, "w+") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
