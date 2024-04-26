import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.cuda.amp import autocast
from tqdm import tqdm
from torchvision import transforms

import wandb
from evaluate import evaluate, eval_mc_segementation, flatten_tensor
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


random_split = False
train_dir_img = Path("/home/xinyue/thesis/Pytorch-UNet-master/data/patches/train/data/")
train_dir_mask = Path("/home/xinyue/thesis/Pytorch-UNet-master/data/patches/train/label/")
val_dir_img = Path("/home/xinyue/thesis/Pytorch-UNet-master/data/patches/validation/data/")
val_dir_mask = Path("/home/xinyue/thesis/Pytorch-UNet-master/data/patches/validation/label/")

dir_checkpoint = Path("/home/xinyue/thesis/Pytorch-UNet-master/checkpoints/")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),    
    transforms.RandomRotation(90),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(1, 1)),  
])

def train_model(
        model,
        device,
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 0.1
):

    # Best Loss initialization
    best_val_loss = float('inf')
    best_val_IoU = [0] * model.n_classes
    best_train_loss = float('inf')
    
    # Given data volume distribution
    data_distribution = {0: 28341786, 1: 4387121, 2: 401287, 3: 1067713, 4: 3476491, 5: 1072859, 6: 115591}
    total_samples = sum(data_distribution.values())
    class_weights = {class_idx: total_samples / num_samples for class_idx, num_samples in data_distribution.items()}
    class_weights[6] = class_weights[6]*2
    class_weights_tensor = torch.tensor(list(class_weights.values()))
    class_weights_tensor = class_weights_tensor.to(device)
    #########################################################
    # Given the training set and validation set respectively
    #########################################################
    # # 1. Create dataset
    try:
         train_set = CarvanaDataset(train_dir_img, train_dir_mask, img_scale, transform=None)
         val_set = CarvanaDataset(val_dir_img, val_dir_mask, img_scale, transform=None)
    except (AssertionError, RuntimeError, IndexError):
         train_set = BasicDataset(train_dir_img, train_dir_mask, img_scale, transform=None)
         val_set = BasicDataset(val_dir_img, val_dir_mask, img_scale, transform=None)
    
    n_train = int(len(train_set))
    n_val = int(len(val_set))
    dataset = ConcatDataset([train_set, val_set])
    ####################################################
    # Random assignment of training and validation sets
    ####################################################
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    # criterion = nn.NLLLoss2d()
    global_step = 0
    epoch_loss = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
    
        epoch_loss = 0
        train_dice_score = 0
        

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:   # Reference to the preprocess function (normalisation) here through Dataloader
                images, true_masks = batch['image'], batch['mask'] # the (batchsize) images to one tensor

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                # data to GPU/CPU
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                # prediction
                masks_pred = model(images)
                
                # training evaluation
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                
                # whether to use one_hot method to give the label to pred_result
                train_IoU_per_class, mean_iou = eval_mc_segementation(
                    masks_pred, true_masks, model, device, train_dice_score, onehot=False
                    )
                
                # optimize training parameters
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                 
                try:
                            experiment.log({
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred[0].argmax(dim=0).float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                            })
                except:
                    pass
                
            # train loss
            epoch_loss /= len(train_loader)

            # val_loss
            val_loss, IoU_per_class, val_MIoU = evaluate(model, val_loader, device, amp)
            scheduler.step(epoch_loss)
            
            
            val_IoU_values = IoU_per_class.cpu().numpy().tolist()
            train_IoU_values = train_IoU_per_class.cpu().numpy().tolist()
            val_loss_values = val_loss.cpu().numpy().tolist()
            
            logging.info(f'train_MIoU_per_class: {train_IoU_values}')
            logging.info(f'val_MIoU_per_class: {val_IoU_values}')
            
            # Update pbar with epoch loss and validation loss
            pbar.set_postfix(**{'epoch_loss': epoch_loss, 'val_loss': val_loss_values})
            # record training process
            experiment.log({
                'train_loss': epoch_loss,
                'val_loss': val_loss_values,
                'step': global_step,
                'epoch': epoch
                })
    
    
        if val_loss < best_val_loss:
            best_val_loss =val_loss_values
            best_val_IoU = val_IoU_values
            best_train_loss = epoch_loss
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                if random_split == True:
                    state_dict['mask_values'] = dataset.mask_values
                else:
                    try:
                        state_dict['mask_values'] = train_set.mask_values
                    except (AttributeError):
                        state_dict['mask_values'] = val_set.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch.pth'))
                logging.info(f'Checkpoint {epoch} saved!')
    
    logging.info(f'best val IoU: {best_val_IoU}')
    logging.info(f'best val loss: {best_val_loss}')
    logging.info(f'best train loss: {best_train_loss}')
                
    experiment.finish()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=7, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
     
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

    except MemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
