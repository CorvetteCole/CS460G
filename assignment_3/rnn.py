# Import necessary libraries
import os
import time
import random
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Detect if GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)

        self.fc = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x, batch_size):
        output, hidden_state = self.rnn(x, self.init_hidden(batch_size))
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden


def train_model(model, dataloader, loss, optimizer, epochs, vocab_size):
    running_loss = 0
    for epoch in range(epochs):
        running_loss = 0
        start_time = time.time()

        for i, (inputs, targets) in enumerate(dataloader):
            batch_size, seq_len = inputs.size()
            inputs = nn.functional.one_hot(inputs, vocab_size).to(torch.float32).to(device)
            targets = targets.view(-1).long()

            optimizer.zero_grad()
            output, hidden = model(inputs, batch_size)
            loss_value = loss(output, targets)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
        end_time = time.time()

        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {end_time - start_time:.2f} sec')

    return running_loss / len(dataloader)


def predict_next(model, input_str, num_chars, char2int, int2char, vocab_size):
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Convert input string to integer sequence
        input_seq = [char2int[char] for char in input_str]
        input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(device)

        # Use model to predict next characters
        output_str = input_str
        for i in range(num_chars):
            # Convert input tensor to one-hot encoding
            # input_one_hot = create_one_hot(input_tensor.cpu().numpy(), vocab_size)
            # input_one_hot = input_one_hot.to(device)
            input_one_hot = nn.functional.one_hot(input_tensor, vocab_size).to(torch.float32).to(device)

            # Make prediction
            output, hidden = model(input_one_hot, 1)
            output = nn.functional.softmax(output[-1], dim=0).data

            # Sample next character based on probability distribution
            next_char_index = torch.multinomial(output, 1).item()
            next_char = int2char[next_char_index]
            output_str += next_char

            # Update input tensor with next character
            input_tensor = torch.cat((input_tensor[:, 1:], torch.LongTensor([[next_char_index]]).to(device)), dim=1).to(device)

        return output_str


def preprocess_data(data_file, batch_size):
    # Read in data and preprocess
    sentences = []
    with open(data_file, 'r') as f:
        for line in f:
            if line != '\n':
                sentences.append(line.strip())

    # Extract all characters
    characters = set(''.join(sentences))

    # Set up the vocabulary
    int2char = dict(enumerate(characters))
    char2int = {character: index for index, character in int2char.items()}
    vocab_size = len(char2int)

    # Prepare input and target sequences
    input_sequence = [sentence[:-1] for sentence in sentences]
    target_sequence = [sentence[1:] for sentence in sentences]

    # Convert sequences to integers
    input_sequence = [[char2int[character] for character in sentence] for sentence in input_sequence]
    target_sequence = [[char2int[character] for character in sentence] for sentence in target_sequence]

    # Pad sequences to equal length
    max_length = max(len(sequence) for sequence in input_sequence)
    input_sequence = [sequence + [0] * (max_length - len(sequence)) for sequence in input_sequence]
    target_sequence = [sequence + [0] * (max_length - len(sequence)) for sequence in target_sequence]

    # Convert sequences to tensors
    input_sequence = torch.LongTensor(input_sequence).to(device)
    target_sequence = torch.LongTensor(target_sequence).to(device)

    # Create DataLoader to generate batches
    dataset = TensorDataset(input_sequence, target_sequence)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader, vocab_size, char2int, int2char, max_length


def main(epochs, batch_size, hidden_size, num_layers, input_model_file, output_model_file, resume_training):
    dataloader, vocab_size, char2int, int2char, max_length = preprocess_data('data/tiny-shakespeare.txt', batch_size)

    # Train the model with batches
    model = RNNModel(vocab_size, vocab_size, hidden_size, num_layers)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    should_train = True
    # check if file model_save_name exists, if so prompt the user if they want to load the model
    if input_model_file is not None and os.path.isfile(input_model_file):
        print(f'Found model: {input_model_file}')
        model.load_state_dict(torch.load(input_model_file))
        print('Evaluating loaded model...')

        model.eval()
        with torch.no_grad():
            pretrained_loss = 0
            for i, (inputs, targets) in enumerate(dataloader):
                batch_size, seq_len = inputs.size()
                inputs = nn.functional.one_hot(inputs, vocab_size).to(torch.float32).to(device)
                targets = targets.view(-1).long()

                output, hidden = model(inputs, batch_size)
                loss_value = loss(output, targets)
                pretrained_loss += loss_value.item()
        print(f'Loaded model loss: {pretrained_loss / len(dataloader):.4f}')

        # return pretrained_loss / len(dataloader)

        if resume_training or input(f'Continue training model ({input_model_file})? [y/N]: ').lower().strip() == 'y':
            model.train()
        else:
            should_train = False

    if should_train:
        train_model(model, dataloader, loss, optimizer, epochs, vocab_size)

    # Save model
    if output_model_file is not None:
        torch.save(model.state_dict(), output_model_file)

    while True:
        try:
            num_predictions = int(input(f"Enter # of characters to generate: "))
        except ValueError:
            print("Invalid input")
            continue

        start = input(f"Enter a prompt (max length {max_length}): ")

        if len(start) > max_length:
            print(f"Prompt too long, must be less than {max_length} characters")
            continue

        print(predict_next(model, start, num_predictions, char2int, int2char, vocab_size))


def find_optimal_hyperparameters(num_trials):
    search_space = {
        'epochs': [1, 2, 3],
        'batch_size': [8, 16, 32, 64],
        'hidden_size': [100, 200, 300, 400],
        'num_layers': [1, 2, 3]
    }

    def get_loss(batch_size, hidden_size, num_layers, epochs):
        dataloader, vocab_size, char2int, int2char, max_length = preprocess_data('data/tiny-shakespeare.txt',
                                                                                 batch_size)
        model = RNNModel(vocab_size, vocab_size, hidden_size, num_layers)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        return train_model(model, dataloader, loss, optimizer, epochs, vocab_size)

    best_hyperparameters = None
    lowest_loss = float('inf')
    for _ in range(num_trials):
        hyperparameters = {key: random.choice(values) for key, values in search_space.items()}
        print(f'Testing hyperparameters: {hyperparameters}')

        new_loss = get_loss(**hyperparameters)
        if new_loss < lowest_loss:
            lowest_loss = new_loss
            best_hyperparameters = hyperparameters
            print(f'New best hyperparameters: {best_hyperparameters}')

    print(f'Best hyperparameters: {best_hyperparameters}')
    print(f'Lowest loss: {lowest_loss}')


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    # training args
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-s', '--hidden_size', type=int, default=100)
    parser.add_argument('-l', '--num_layers', type=int, default=1)
    # loading args
    parser.add_argument('-i', '--input_model', type=str, default='bad_rnn.ckpt')
    parser.add_argument('-o', '--output_model', type=str, default='bad_rnn.ckpt')
    parser.add_argument('-r', '--resume_training', type=bool, default=False)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.hidden_size, args.num_layers, args.input_model, args.output_model,
         args.resume_training)
    # find_optimal_hyperparameters(15)
