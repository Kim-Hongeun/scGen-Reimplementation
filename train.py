import os
import argparse
import numpy as np
import torch
from torch import optim, nn
from scgenVAE import Encoder, Decoder, scgenVAE
from dataloader import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)   

    # Build data loader
    train_loader = get_loader(args.train_path, args.batch_size,
                             shuffle=True)
    test_loader = get_loader(args.test_path, args.batch_size,
                             shuffle=True) 
                        


    # Build the models
    encoder = Encoder(args.n_input, args.n_hidden1, args.n_hidden2, args.n_latent)
    decoder = Decoder(args.n_latent, args.n_hidden2, args.n_hidden1, args.n_output)

    model = scgenVAE(Encoder = encoder, Decoder = decoder).to(device)


    # Loss and optimizer
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for batch_idx, x in enumerate(train_loader):
            
            # Forward, backward and optimize
            x_hat, mean, var = model(x)
            loss = model.loss_function(x_hat, mean, var)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if batch_idx % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, batch_idx, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (batch_idx+1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'scGen-{}-{}.ckpt'.format(epoch+1, batch_idx+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--train_path', type=str, default='data/train_pbmc.h5ad', help='directory for resized images')
    parser.add_argument('--test_path', type=str, default='data/valid_pbmc.h5ad', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--n_input', type=int , default=7002, help='dimension of input vectors')
    parser.add_argument('--n_hidden1', type=int , default=1024, help='dimension of first hidden states')
    parser.add_argument('--n_hidden2', type=int , default=512, help='dimension of second hidden states')
    parser.add_argument('--n_latent', type=int , default=100, help='dimension of latent vecotrs')
    parser.add_argument('--n_output', type=int , default=7002, help='dimension of output vectors')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
