from __future__ import absolute_import, print_function, division
from datetime import datetime
import numpy as np
import torch
import time


def save_checkpoint(checkpoint_file, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer': optimizer_state_dict}
    torch.save(states, checkpoint_file)


def load_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch']
    model_state_dict, optimizer_state_dict = checkpoint['state_dict'], checkpoint['optimizer']
    return epoch, model_state_dict, optimizer_state_dict


def current_learning_rate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']


def current_weight_decay(optimizer):
    return optimizer.state_dict()['param_groups'][0]['weight_decay']


########################################################################################################################
#####################################################   Training   #####################################################
########################################################################################################################


def train_epoch(model, criterion, optimizer, device, data_loader,
                epoch: int, batch_size: int, summary_writer=None):
    print('#' * 50 + '  TRAINING epoch {}  '.format(epoch+1) + '#' * 50)
    print('Epoch {}. Starting with training phase.'.format(epoch + 1))

    model.train()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    # accuracies = np.zeros(steps_in_epoch, dtype=np.float32)

    example_idx = 0
    log_frequency = 5
    log_image_frequency = 5
    print_frequency = 5
    epoch_start_time = time.time()
    for step, (x, y) in enumerate(data_loader):
        start_time = time.time()

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        x = x.to(device)
        y = y.to(device)
        # y = torch.unsqueeze(y, -1)

        # Feed-forward through the network
        y_hat = model.forward(x)

        # Calculate loss
        loss = criterion(y_hat, y)

        # # Calculate accuracy
        # correct = torch.sum((torch.abs(y_hat - y) < torch.abs(percentage_close * y)))
        # accuracy = correct.double() / y.size

        # Calculate elapsed time for this step
        examples_per_second = batch_size / float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Save statistics
        losses[step] = loss.item()
        # accuracies[step] = accuracy.item()

        # Print statistics
        if step % print_frequency == 0 and step != 0:
            print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.4f}, Loss = {:.3f}".format(  #, Accuracy = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch+1,
                step, steps_in_epoch, examples_per_second,
                current_learning_rate(optimizer),
                losses[step]))  #, np.mean(accuracies[step-print_frequency: step])))

        # Log statistics
        if summary_writer:
            if step % log_frequency == 0:
                global_step = (epoch * steps_in_epoch) + step  # compute the global step, only for logging
                summary_writer.add_scalar('train/loss', losses[step], global_step)
                # summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
                summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
                summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
                summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)
            if step % log_image_frequency == 0:
                clip_for_display = x[example_idx].clone().cpu()
                min_val = float(clip_for_display.min())
                max_val = float(clip_for_display.max())
                clip_for_display.clamp_(min=min_val, max=max_val)
                clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    # epoch_avg_acc = float(np.mean(accuracies))

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        # summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_duration, epoch_avg_loss  #, epoch_avg_acc


########################################################################################################################
####################################################   Validation   ####################################################
########################################################################################################################


def validation_epoch(model, criterion, device, data_loader,
                     epoch: int, batch_size: int, summary_writer=None):
    print('#' * 50 + '  VALIDATION epoch {}  '.format(epoch+1) + '#' * 50)
    print('Epoch {}. Starting with validation phase.'.format(epoch + 1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    # accuracies = np.zeros(steps_in_epoch, dtype=np.float32)

    example_idx = 0
    print_frequency = 3
    epoch_start_time = time.time()
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            start_time = time.time()

            # Move inputs to GPU memory
            x = x.to(device)
            y = y.to(device)
            # y = torch.unsqueeze(y, -1)

            # Feed-forward through the network
            y_hat = model.forward(x)

            # Calculate loss
            loss = criterion(y_hat, y)

            # # Calculate accuracy
            # correct = torch.sum((torch.abs(y_hat - y) < torch.abs(percentage_close * y)))
            # accuracy = correct.double() / y.size

            # Calculate elapsed time for this step
            examples_per_second = batch_size / float(time.time() - start_time)

            # Save statistics
            losses[step] = loss.item()
            # accuracies[step] = accuracy.item()

            if step % print_frequency == 0 and step > 0:
                print("[{}] Epoch {}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Loss = {:.3f}".format(  #, Accuracy = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    losses[step]))  #, np.mean(accuracies[step-print_frequency: step])))

            if summary_writer and step == 0:
                clip_for_display = x[example_idx].clone().cpu()
                min_val = float(clip_for_display.min())
                max_val = float(clip_for_display.max())
                clip_for_display.clamp_(min=min_val, max=max_val)
                clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    # epoch_avg_acc = float(np.mean(accuracies))

    return epoch_duration, epoch_avg_loss  #, epoch_avg_acc


########################################################################################################################
######################################################   Testing   #####################################################
########################################################################################################################


def testing_epoch(model, criterion, device, data_loader,
                  epoch: int, batch_size: int, summary_writer=None):
    print('#' * 50 + '   TESTING   ' + '#' * 50)
    print('Starting with final testing phase.')

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    # accuracies = np.zeros(steps_in_epoch, dtype=np.float32)

    targets, predictions = [], []

    example_idx = 0
    log_frequency = 3
    log_image_frequency = 6
    print_frequency = 3
    epoch_start_time = time.time()
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            start_time = time.time()

            # Move inputs to GPU memory
            x = x.to(device)
            y = y.to(device)
            # y = torch.unsqueeze(y, -1)

            # Feed-forward through the network
            y_hat = model.forward(x)
            predictions.append(y_hat)
            targets.append(y)

            # Calculate loss
            loss = criterion(y_hat, y)

            # # Calculate accuracy
            # correct = torch.sum((torch.abs(y_hat - y) < torch.abs(percentage_close * y)))
            # accuracy = correct.double() / y.size

            # Calculate elapsed time for this step
            examples_per_second = batch_size / float(time.time() - start_time)

            # Save statistics
            losses[step] = loss.item()
            # accuracies[step] = accuracy.item()

            # Print statistics
            if step % print_frequency == 0 and step > 0:
                print("[{}] Testing Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Loss = {:.3f}".format(  #, Accuracy = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"),
                    step, steps_in_epoch, examples_per_second,
                    losses[step]))  #, np.mean(accuracies[step-print_frequency: step])))

            # Log statistics
            if summary_writer:
                if step % log_frequency == 0:
                    global_step = (epoch * steps_in_epoch) + step  # compute the global step, only for logging
                    summary_writer.add_scalar('test/loss', losses[step], global_step)
                    # summary_writer.add_scalar('test/accuracy', accuracies[step], global_step)
                    summary_writer.add_scalar('test/examples_per_second', examples_per_second, global_step)
                if step % log_image_frequency == 0:
                    clip_for_display = x[example_idx].clone().cpu()
                    min_val = float(clip_for_display.min())
                    max_val = float(clip_for_display.max())
                    clip_for_display.clamp_(min=min_val, max=max_val)
                    clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    # epoch_avg_acc = float(np.mean(accuracies))

    if summary_writer:
        summary_writer.add_scalar('test/epoch_avg_loss', epoch_avg_loss, epoch)
        # summary_writer.add_scalar('test/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_duration, epoch_avg_loss, (targets, predictions)  #, epoch_avg_acc
