#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

from torch import rand, Tensor, cat, zeros
from torch.cuda import is_available
from torch.nn import GRU, LSTM, Linear, MSELoss, Sigmoid, \
    ReLU, GRUCell, BCEWithLogitsLoss
from torch.optim import Adam

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = []


def rnns_as_black_box_gru():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 2
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 20
    in_features = 8
    out_features = 2

    rnn = GRU(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True).to(device)
    linear = Linear(in_features=4, out_features=out_features).to(device)
    activation = Sigmoid()
    loss_f = MSELoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    x = rand(nb_examples, t_steps, in_features).to(device)
    y = rand(nb_examples, t_steps, out_features).to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i+batch_size, :, :]
            y_out = y[i:i+batch_size, :, :]
            y_hat = activation(linear(rnn(x_in)[0]))

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def rnns_as_black_box_lstm():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 2
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 20
    in_features = 8
    out_features = 2

    rnn = LSTM(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True).to(device)
    linear = Linear(in_features=4, out_features=out_features).to(device)
    activation = Sigmoid()
    loss_f = MSELoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    x = rand(nb_examples, t_steps, in_features).to(device)
    y = rand(nb_examples, t_steps, out_features).to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i+batch_size, :, :]
            y_out = y[i:i+batch_size, :, :]
            y_hat = activation(linear(rnn(x_in)[0]))

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def rnn_iteration():
    device = 'cuda' if is_available() else 'cpu'
    print(f'Using {device}')
    epochs = 128
    batch_size = 8
    all_examples = batch_size * 32
    sequence_length = 64

    input_features = 8
    hidden_size_1 = 4
    hidden_size_2 = 2
    output_features = 2

    gru_cell_1 = GRUCell(input_size=input_features, hidden_size=hidden_size_1).to(device)
    gru_cell_2 = GRUCell(input_size=hidden_size_1, hidden_size=hidden_size_2).to(device)
    linear = Linear(in_features=hidden_size_2, out_features=output_features).to(device)

    relu = ReLU()
    loss_function = MSELoss()

    optim = Adam(chain(
        gru_cell_1.parameters(),
        gru_cell_2.parameters(),
        linear.parameters()))

    x = rand(all_examples, sequence_length, input_features).to(device)
    y = rand(all_examples, sequence_length, output_features).to(device)

    for epoch in range(epochs):

        epoch_loss = []

        for b_i in range(0, x.size()[0], batch_size):
            optim.zero_grad()

            x_batch = x[b_i:b_i+batch_size, :, :]
            y_batch = y[b_i:b_i+batch_size, :, :]

            h_1 = zeros(batch_size, hidden_size_1).to(device)
            h_2 = zeros(batch_size, hidden_size_2).to(device)

            y_hat = []

            for t_step in range(x_batch.size()[1]):
                h_1 = gru_cell_1(x_batch[:, t_step, :], h_1)
                h_2 = gru_cell_2(h_1, h_2)
                out = relu(linear(h_2))
                y_hat.append(out.unsqueeze(1))

            y_hat = cat(y_hat, dim=1)
            loss = loss_function(y_hat, y_batch)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def k_hot_encoding_case():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 2
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 20
    in_features = 8
    out_features = 4

    rnn = GRU(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True).to(device)
    linear = Linear(in_features=4, out_features=out_features).to(device)
    loss_f = BCEWithLogitsLoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    x = rand(nb_examples, t_steps, in_features).to(device)
    y = rand(nb_examples, t_steps, out_features).ge(0.5).float().to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i+batch_size, :, :]
            y_out = y[i:i+batch_size, :, :]
            y_hat = linear(rnn(x_in)[0])

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def main():
    print('Running GRU case')
    rnns_as_black_box_gru()
    print('-' * 100)
    print('Running LSTM case')
    rnns_as_black_box_lstm()
    print('-' * 100)
    print('-' * 100)
    print('-' * 100)
    print('Running GRUCell case')
    rnn_iteration()
    print('-' * 100)
    print('-' * 100)
    print('Running k-hot encoding case')
    k_hot_encoding_case()


if __name__ == '__main__':
    main()

# EOF
