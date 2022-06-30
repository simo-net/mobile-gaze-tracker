import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import argparse
import json, pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from gazetracker.dataset.loader import Gaze_Capture
from gazetracker.models.gazetrack import gazetrack_model
from gazetracker.utils.runner import NoGPUError, train, test

# Usage:
# python ./train-test-gazetrack.py  -data_dir ~/data/..  -log_dir ./logs/..  -batch_size 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the training set')
    parser.add_argument('-log_dir', type=str)
    parser.add_argument('-batch_size', type=int, default=5)
    parser.add_argument('-patience', type=int, default=6)
    parser.add_argument('-max_num_epochs', type=int, default=50)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-lr_sched_patience', type=int, default=5)
    parser.add_argument('-checkpoint_frequency', type=int, default=None)
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-workers', type=int, default=4)
    return parser.parse_args()


def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()

    # Check if GPU is available --------------------------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise NoGPUError('No torch-compatible GPU found. Aborting.')

    # Load training dataset ------------------------------------------------------------------------#
    train_dataset = Gaze_Capture(os.path.join(args.data_dir, "train"), split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_dataset = Gaze_Capture(os.path.join(args.data_dir, "val"), split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # Build the model ------------------------------------------------------------------------------#
    model = gazetrack_model()
    if device.type == 'cuda':
        model.cuda()
        model = nn.DataParallel(model)

    # Training -------------------------------------------------------------------------------------#
    _, history = train(model, train_loader, val_loader,
                       checkpoint_dir=args.log_dir, checkpoint_frequency=args.checkpoint_frequency,
                       max_num_epochs=args.max_num_epochs, batch_size=args.batch_size, patience=args.patience,
                       optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                       lr_scheduler='plateau', lr_sched_patience=args.lr_sched_patience,
                       device=device, seed=args.seed)
    # Store training history
    history_file = os.path.join(args.log_dir, 'history')
    with open(history_file, 'wb') as outfile:
        pickle.dump(history, outfile)  # Read as: history = pickle.load(open(history_file, "rb"))

    # Load testing dataset -------------------------------------------------------------------------#
    del train_dataset, val_dataset, train_loader, val_loader
    test_dataset = Gaze_Capture(os.path.join(args.data_dir, "test"), split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

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
