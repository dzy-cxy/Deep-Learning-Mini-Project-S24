import os
import data  # Assuming this is your module for data loading
import models  # This contains the ResNetModel
import options  # Module for parsing command line arguments
import torch
from torchsummary import summary

def train(opt, model):
    print(opt)
    # Ensure the checkpoint directory exists
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    
    with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
        f.write(str(opt) + '\n')
    
    # Load data
    dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    
    # Initialize the model
    model = models.ResNetModel(opt, train=True)

    total_iter = 0
    loss = 0.0

    while True:
        for batch in dataloader:
            total_iter += 1
            inputs, labels = batch
            model.optimize_params(inputs, labels)
            loss += model.get_current_loss()
            if total_iter % 400 == 0:
                model.scheduler.step()


            # Print and log the loss
            if total_iter % opt.print_freq == 0:
                avg_loss = loss / opt.print_freq
                print(f'Iter: {total_iter:6d}, Loss: {avg_loss:.4f}')
                with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
                    f.write(f'Iter: {total_iter:6d}, Loss: {avg_loss:.4f}\n')
                loss = 0.0  # Reset the loss for the next set of iterations

            # Save model parameters periodically
            if total_iter % opt.save_params_freq == 0:
                model.save_model(f'{total_iter}')

            # Check if the training should stop
            if total_iter >= opt.n_iter:
                model.save_model('final')
                print("Training complete.")
                return

if __name__ == '__main__':
    args = options.parse_args_train()  # Ensure this function exists and correctly parses needed options
    model = models.ResNetModel(args, train=True)
    
    # Call summary right after model initialization, before entering the training loop
    summary(model.net, input_size=(3, 32, 32), device=str(model.device))

    train(args, model)
