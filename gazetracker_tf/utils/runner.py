import tensorflow as tf


class NoGPUError(Exception):
    pass


def train(model, training_set, validation_set,
          max_num_epochs: int, steps_per_epoch: int = None, validation_steps: int = None,
          patience: int = 10, lr_init: float = 0.016,
          checkpoint_file: str = '../logs/checkpoint'):
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='mse', metrics=["loss"])

    # Build the early-stopping and checkpoint callbacks
    early_stopping = True if 0 < patience < max_num_epochs else False
    es_callback = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min',
        patience=patience, restore_best_weights=True,
        verbose=True), ]
    cp_callback = [tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min',
        filepath=checkpoint_file,
        save_weights_only=True, save_best_only=True, save_freq='epoch',
        verbose=True), ]
    lr_callback = [tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_init,
        decay_steps=8000, decay_rate=0.64,
        staircase=True), ]

    # Train the model
    history = model.fit(training_set, validation_data=validation_set,
                        epochs=max_num_epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=cp_callback + lr_callback + (es_callback if early_stopping else []),
                        verbose=True)
    return model, history.history


def test(model, test_set,
         evaluation_steps: int):
    results = model.evaluate(test_set,
                             steps=evaluation_steps,
                             verbose=True)
    res_dict = {}
    print('Performance:')
    for name, value in zip(model.metrics_names, results):
        print(f'   {name} -> {value}')
        res_dict[name] = value
    return res_dict


def find_best_epoch(history, mode='min_loss'):
    if mode == 'min_loss':
        val_loss = history['val_loss']
        best_epoch = val_loss.index(min(val_loss))  # + 1
    elif mode == 'max_accuracy':
        val_acc = history['val_accuracy']
        best_epoch = val_acc.index(max(val_acc))    # + 1
    else:
        raise ValueError('The mode can either be "min_loss" or "max_accuracy".')
    return best_epoch


def load_validation_best(history, best_epoch: int):
    best_accuracy = history['val_accuracy'][best_epoch]
    best_loss = history['val_loss'][best_epoch]
    return best_accuracy, best_loss
