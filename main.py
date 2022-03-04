import os
import argparse
import numpy as np
import torch
from torch import optim, nn
from scgenVAE import Encoder, Decoder, scgenVAE
from dataloader import get_loader


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models' , help='path for saving trained models')
parser.add_argument('--train_path', type=str, default='data/train_pbmc.h5ad', help='directory for resized images')
parser.add_argument('--test_path', type=str, default='data/valid_pbmc.h5ad', help='directory for resized images')
parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int , default=200, help='step size for saving trained models')

# Model parameters
parser.add_argument('--n_input', type=int , default=7000, help='dimension of input vectors')
parser.add_argument('--n_hidden1', type=int , default=1024, help='dimension of first hidden states')
parser.add_argument('--n_hidden2', type=int , default=512, help='dimension of second hidden states')
parser.add_argument('--n_latent', type=int , default=256, help='dimension of latent vecotrs')
parser.add_argument('--n_output', type=int , default=7000, help='dimension of output vectors')

parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=0.0001)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model directory
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)   

# Build data loader
train_loader = get_loader(args.train_path, args.batch_size,
                            shuffle=True)
test_loader = get_loader(args.test_path, args.batch_size,
                            shuffle=False) 

# Build the models
encoder = Encoder(args.n_input, args.n_hidden1, args.n_hidden2, args.n_latent)
decoder = Decoder(args.n_latent, args.n_hidden2, args.n_hidden1, args.n_output)

model = scgenVAE(Encoder = encoder, Decoder = decoder).to(device)

# Loss and optimizer
params = list(model.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, x in enumerate(train_loader):      
        # Forward, backward and optimize\
        x = x.to(device)
        x_hat, mean, var = model(x)
        loss = model.loss_function(x_hat, x, mean, var)
        model.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Print log info
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    # Save the model checkpoints
    if (epoch+1) % args.save_step == 0:
        torch.save(model.state_dict(), os.path.join(
            args.model_path, 'scGen-Feb22-{}.ckpt'.format(epoch+1)))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            x_hat, mean, var = model(x)
            test_loss += model.loss_function(x_hat, x, mean, var).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()