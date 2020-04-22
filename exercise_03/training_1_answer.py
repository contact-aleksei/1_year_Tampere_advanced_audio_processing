#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from copy import deepcopy

from torch import no_grad, Tensor
from torch.cuda import is_available
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from getting_and_init_the_data_answer import get_data_loader, get_dataset
from my_cnn_system_answer import MyCNNSystem

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = []


def main():

    device = 'cuda' if is_available() else 'cpu'
    print(f'\nProcess on {device}\n')
    data_path = Path('music_speech_dataset')
    batch_size = 4
    epochs = 300

    cnn_1_channels = 16
    cnn_1_kernel = 5
    cnn_1_stride = 2
    cnn_1_padding = 2
    # Output size: b_size x 16 x 323 x 20

    pooling_1_kernel = 3
    pooling_1_stride = 1
    # Output size: b_size x 16 x 321 x 18

    cnn_2_channels = 32
    cnn_2_kernel = 5
    cnn_2_stride = 2
    cnn_2_padding = 2
    # Output size: b_size x 32 x 161 x 9

    pooling_2_kernel = 3
    pooling_2_stride = 2
    # Output size: b_size x 32 x 80 x 40

    classifier_input = 10240
    classifier_output = 1

    dropout = .25

    loader_training = get_data_loader(
        dataset=get_dataset('training', data_path),
        batch_size=batch_size,
        shuffle=True)

    loader_validation = get_data_loader(
        dataset=get_dataset('validation', data_path),
        batch_size=batch_size,
        shuffle=True)

    loader_testing = get_data_loader(
        dataset=get_dataset('testing', data_path),
        batch_size=batch_size,
        shuffle=False)

    model = MyCNNSystem(
        cnn_channels_out_1=cnn_1_channels,
        cnn_kernel_1=cnn_1_kernel,
        cnn_stride_1=cnn_1_stride,
        cnn_padding_1=cnn_1_padding,
        pooling_kernel_1=pooling_1_kernel,
        pooling_stride_1=pooling_1_stride,
        cnn_channels_out_2=cnn_2_channels,
        cnn_kernel_2=cnn_2_kernel,
        cnn_stride_2=cnn_2_stride,
        cnn_padding_2=cnn_2_padding,
        pooling_kernel_2=pooling_2_kernel,
        pooling_stride_2=pooling_2_stride,
        classifier_input_features=classifier_input,
        output_classes=classifier_output,
        dropout=dropout)

    model = model.to(device)

    optimizer = Adam(model.parameters())

    loss_function = BCEWithLogitsLoss()

    min_epoch_loss = 1e10
    patience = 30
    patience_counter = 0

    best_model = None

    for epoch in range(epochs):

        loss_training = []
        loss_validation = []

        model.train(True)
        for training_data in loader_training:
            optimizer.zero_grad()

            x, y = training_data

            x = x.to(device)
            y = y.to(device).float()

            y_hat = model(x).squeeze(1)

            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_training.append(loss.item())

        model.eval()
        with no_grad():
            for validation_data in loader_validation:

                x, y = validation_data

                x = x.to(device)
                y = y.to(device).float()

                y_hat = model(x).squeeze(1)

                loss = loss_function(y_hat, y)

                loss_validation.append(loss.item())

        loss_training_mean = Tensor(loss_training).mean()
        loss_validation_mean = Tensor(loss_validation).mean()

        if loss_validation_mean < min_epoch_loss:
            min_epoch_loss = loss_validation_mean
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly stopping! Best validation loss: {min_epoch_loss:7.4f}', end='\n\n')
            break
        else:
            best_model = deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d} | '
              f'Training loss: {loss_training_mean:7.4f} | '
              f'Validation loss: {loss_validation_mean:7.4f}')

    print('Starting testing...', end=' ')

    loss_testing = []

    if best_model is not None:
        model.load_state_dict(best_model)
    else:
        print('No best model found. Exiting')
        exit()

    model.eval()
    with no_grad():
        for testing_data in loader_testing:
            x, y = testing_data

            x = x.to(device)
            y = y.to(device).float()

            y_hat = model(x).squeeze(1)

            loss = loss_function(y_hat, y)

            loss_testing.append(loss.item())

    print(f'loss: {Tensor(loss_testing).mean():7.4f}')


if __name__ == '__main__':
    main()

# EOF
