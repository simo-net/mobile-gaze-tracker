import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import json, pickle
import torch
import torch.nn as nn
from gazetracker.models.gazetrack import GazeTracker
from gazetracker.dataset.loader import GazeCapture, create_dataloader
from gazetracker.utils.runner import NoGPUError, train, test

# Usage:
# python ./train-test-gazetrack.py  -data_dir ~/data/..  -log_dir ./logs/..  -batch_size 20


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-lr_sched_patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility. Default is None.')
    return parser.parse_args()


def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()

    # Check if GPU is available --------------------------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise NoGPUError('No torch-compatible GPU found. Aborting.')

    # Load training dataset ------------------------------------------------------------------------#
    train_dataset = GazeCapture(root=os.path.join(args.data_dir, "train"), size=(128, 128), verbose=True)
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = GazeCapture(root=os.path.join(args.data_dir, "val"), size=(128, 128), verbose=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    # NUM_TRAIN_SAMPLES = len(train_dataset)
    # NUM_VAL_SAMPLES = len(val_dataset)
    # num_train_samples = NUM_TRAIN_SAMPLES - NUM_VAL_SAMPLES
    # num_train_batches = num_train_samples // args.batch_size
    # num_val_batches = NUM_VAL_SAMPLES // args.batch_size

    # Build the model ------------------------------------------------------------------------------#
    model = GazeTracker()
    if device.type == 'cuda':
        model.cuda()
        model = nn.DataParallel(model)

    # Training -------------------------------------------------------------------------------------#
    _, history = train(model, train_loader, val_loader,
                       checkpoint_dir=args.log_dir, checkpoint_frequency=None,
                       batch_size=args.batch_size, max_num_epochs=args.max_num_epochs, patience=args.patience,
                       lr=args.lr_init, betas=(0.9, 0.999), epsilon=1e-7,
                       lr_scheduler='plateau', lr_sched_patience=args.lr_sched_patience,
                       device=device, seed=args.seed)
    # Store training history
    history_file = os.path.join(args.log_dir, 'history')
    with open(history_file, 'wb') as outfile:
        pickle.dump(history, outfile)  # Read as: history = pickle.load(open(history_file, "rb"))

    # Load testing dataset -------------------------------------------------------------------------#
    del train_dataset, val_dataset, train_loader, val_loader
    test_dataset = GazeCapture(root=os.path.join(args.data_dir, "test"), size=(128, 128), verbose=True)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    # NUM_TEST_SAMPLES = len(test_dataset)
    # num_test_batches = NUM_TEST_SAMPLES // args.batch_size

    # Testing --------------------------------------------------------------------------------------#
    test_loss, [targets, predictions] = test(model, test_loader,
                                             checkpoint_dir=args.log_dir,
                                             batch_size=args.batch_size,
                                             device=device)
    # Store the test results
    results = {'loss': test_loss}  #, 'accuracy': test_acc}
    eval_file = os.path.join(args.log_dir, 'evaluation.json')
    with open(eval_file, "w+") as f:
        json.dump(results, f, indent=2)
    # TODO: save both targets and predictions !!!


if __name__ == '__main__':
    main()
