#!/usr/bin/env python
# -*- coding: utf-8 -*-
from my_cnn_system_aleksei_sapozhnikov import MyCNNSystem
from getting_and_init_the_data_aleksei_sapozhnikov import mydatasets
from copy import deepcopy
import numpy as np
from torch import cuda, no_grad
from torch.nn import  MSELoss
from torch.optim import Adam


# Check if CUDA is available, else use CPU
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Process on {device}', end='\n\n')

# Define hyper-parameters to be used.
epochs = 100
dropout = 0.2
nb_examples_training = 60
nb_examples_validation = 20
nb_examples_testing = 20
batch_size = 2

# Instantiate our DNN
example_dnn = MyCNNSystem(kernel_size_1=2, kernel_size_2=2, channel_1=20, channel_2=20, dropout=0.2)

# Pass DNN to the available device.
example_dnn = example_dnn.to(device) # was commented

# Give the parameters of our DNN to an optimizer.
optimizer = Adam(params=example_dnn.parameters(), lr=1e-3)

# Instantiate our loss function as a class.
loss_function = MSELoss()

# Create our training dataset.
# Create our validation dataset.
# Create our testing dataset.
training, testing, validation = mydatasets()

# Variables for the early stopping
lowest_validation_loss = 1e10
best_validation_epoch = 0
patience = 30
patience_counter = 0

best_model = None

# Start training.
for epoch in range(epochs):

    # Lists to hold the corresponding losses of each epoch.
    epoch_loss_training = []
    epoch_loss_validation = []

    # Indicate that we are in training mode, so (e.g.) dropout
    # will function
    example_dnn.train()

    # For each batch of our dataset.
    for i in training:
        # Zero the gradient of the optimizer.
        optimizer.zero_grad()

        # Get the batches.
        x_input = i[0]
        y_output = i[1].view(2, -1)

        # Get the predictions of our model.
        y_hat = example_dnn(x_input)

        # Calculate the loss of our model.
        loss = loss_function(input=y_hat, target=y_output)

        # Do the backward pass
        loss.backward()

        # Do an update of the weights (i.e. a step of the optimizer)
        optimizer.step()

        # Loss the loss of the batch
        epoch_loss_training.append(loss.item())

    # Indicate that we are in training mode, so (e.g.) dropout
    # will **not** function
    example_dnn.eval()

    # Say to PyTorch not to calculate gradients, so everything will
    # be faster.
    with no_grad():

        # For every batch of our validation data.
        for i in validation:

            # Get the batch
            x_input = i[0]
            y_output = i[1].view(2, -1)

            # Pass the data to the appropriate device.
#            x_1_input = x_1_input.to(device) # was commented
#            x_2_input = x_2_input.to(device) # was commented
#            y_output = y_output.to(device) # was commented

            # Get the predictions of the model.
            y_hat = example_dnn(x_input)

            # Calculate the loss.
            loss = loss_function(input=y_hat, target=y_output)

            # Log the validation loss.
            epoch_loss_validation.append(loss.item())

    # Calculate mean losses.
    epoch_loss_validation = np.array(epoch_loss_validation).mean()
    epoch_loss_training = np.array(epoch_loss_training).mean()

    # Check early stopping conditions.
    if epoch_loss_validation < lowest_validation_loss:
        lowest_validation_loss = epoch_loss_validation
        patience_counter = 0
        best_model = deepcopy(example_dnn.state_dict())
        best_validation_epoch = epoch
    else:
        patience_counter += 1

    # If we have to stop, do the testing.
    if patience_counter >= patience:
        print('\nExiting due to early stopping', end='\n\n')
        print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
        if best_model is None:
            print('No best model. ')
        else:
            # Process similar to validation.
            print('Starting testing', end=' | ')
            testing_loss = []
            example_dnn.eval()
            with no_grad():
                for i in testing:
                    x_input = i[0]
                    y_output = i[1].view(2, -1)

#                    x_1_input = x_1_input.to(device) # was commented
#                    x_2_input = x_2_input.to(device) # was commented
#                    y_output = y_output.to(device) # was commented

                    y_hat = example_dnn(x_input)

                    loss = loss_function(input=y_hat, target=y_output)

                    testing_loss.append(loss.item())

            testing_loss = np.array(testing_loss).mean()
            print(f'Testing loss: {testing_loss:7.4f}')
            break
    print(f'Epoch: {epoch:03d} | '
          f'Mean training loss: {epoch_loss_training:7.4f} | '
          f'Mean validation loss {epoch_loss_validation:7.4f}')
