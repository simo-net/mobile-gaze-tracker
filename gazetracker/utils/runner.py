import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from gazetracker.utils.iterators import train_epoch, validation_epoch, testing_epoch, save_checkpoint, load_checkpoint


class NoGPUError(Exception):
    pass


def train(model, train_loader, val_loader,
          checkpoint_dir: str = None,
          checkpoint_frequency: int or None = None,
          batch_size: int = 5,
          patience: int = 5,
          max_num_epochs: int = 50, lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 0.,
          lr_scheduler: str or None = 'plateau', lr_sched_patience: int = 2, optimizer: str = 'sgd',
          device=torch.device('cpu'), seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)

    # configuration options
    writer = SummaryWriter(log_dir=checkpoint_dir) if checkpoint_dir else None
    early_stopping = True if 0 < patience < max_num_epochs else False
    assert optimizer in ['sgd', 'adam'], 'Optimizer can either be "sgd" or "adam".'

    # configure optimizer and criterion
    criterion = F.mse_loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) if optimizer == 'sgd' \
        else optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # configure learning-rate scheduler
    if lr_scheduler == 'plateau':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lr_sched_patience)
    elif lr_scheduler == 'multistep':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_sched_patience, 3*lr_sched_patience])
    else:  # lr_scheduler is None
        lr_sched = None

    # run the training loop for defined number of epochs
    history = {'loss': [], 'val_loss': []}  #, 'accuracy': [], 'val_accuracy': []}
    best_epoch, best_val_acc, best_val_loss = 0, 0., np.inf
    for epoch in range(max_num_epochs):
        # perform one training epoch
        train_duration, train_loss = train_epoch(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    data_loader=train_loader,
                    epoch=epoch,
                    batch_size=batch_size,
                    summary_writer=writer
        )
        history['loss'].append(train_loss)
        # history['accuracy'].append(train_acc)

        # perform one validation epoch
        val_duration, val_loss = validation_epoch(
                    model=model,
                    criterion=criterion,
                    device=device,
                    data_loader=val_loader,
                    epoch=epoch,
                    batch_size=batch_size,
                    summary_writer=writer
        )
        history['val_loss'].append(val_loss)
        # history['val_accuracy'].append(val_acc)

        # update learning rate
        if lr_scheduler == 'plateau':
            lr_sched.step(val_loss)
        elif lr_scheduler == 'multistep':
            lr_sched.step(epoch)

        print('\n' + '-'*50 + f'  EPOCH {epoch+1} SUMMARY  ' + '-'*50)
        print(f'Training Phase.')
        print(f'  Total Duration:         {int(np.ceil(train_duration / 60)) :d} minutes')
        print(f'  Average Train Loss:     {train_loss :.3f}')
        # print(f'  Average Train Accuracy: {train_acc :.3f}\n')

        print('Validation Phase.')
        print(f'  Total Duration:              {int(np.ceil(val_duration / 60)) :d} minutes')
        print(f'  Average Validation Loss:     {val_loss :.3f}')
        # print(f'  Average Validation Accuracy: {val_acc :.3f}\n')

        # frequent model savings
        if checkpoint_dir and checkpoint_frequency and epoch % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch+1 :03d}.pth')
            save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
            print(f'Model checkpoint (periodic) written to:          {checkpoint_path}')

        # saving best model based on min loss
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            print(f'Found new best validation loss: {best_val_loss :.3f}')
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')
                save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
                print(f'Model checkpoint (best loss) written to:         {checkpoint_path}')

        # early stopping
        if early_stopping and epoch >= best_epoch + patience:
            if all([loss < best_val_loss for loss in history['val_loss'][-patience:]]):
                # all last validation losses are bigger than the best one
                print(f'Early stopping because loss on validation set did not decrease in the last {patience} epochs.')
                break

        print('-'*130 + '\n')

    return [float(best_val_loss), best_epoch if early_stopping else max_num_epochs-1], history


def test(model, test_loader,
         checkpoint_dir: str = None,
         batch_size: int = 5,
         device=torch.device('cpu')):

    # configure optimizer and criterion
    criterion = F.mse_loss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0)

    # load training checkpoint
    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise ValueError('Must specify a valid checkpoint directory from where to load pre-trained model weights.')
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')
    check_point = torch.load(checkpoint_path)
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])

    # Perform one validation epoch
    writer = SummaryWriter(log_dir=checkpoint_dir)
    test_duration, test_loss, (targets, predictions) = testing_epoch(
        model=model,
        criterion=criterion,
        device=device,
        data_loader=test_loader,
        epoch=0,
        batch_size=batch_size,
        summary_writer=writer
    )
    print(f'\nTest Phase.')
    print(f'  Total Duration:        {int(np.ceil(test_duration / 60)) :d} minutes')
    print(f'  Average Test Loss:     {test_loss :.3f}')
    # print(f'  Average Test Accuracy: {test_acc :.3f}\n')

    return float(test_loss), (targets, predictions)


def find_best_epoch(history):
    val_loss = history['val_loss']
    best_epoch = val_loss.index(min(val_loss))
    return best_epoch


def find_best_epoch_from_checkpoint(checkpoint_file):
    best_epoch = load_checkpoint(checkpoint_file)[0] - 1
    return best_epoch


def load_validation_best(history, best_epoch: int):
    best_loss = history['val_loss'][best_epoch]
    if 'val_accuracy' in history.keys():
        best_accuracy = history['val_accuracy'][best_epoch]
        return best_accuracy, best_loss
    return best_loss
