import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.data_loading import check_pixel_values


def flatten_tensor(tensor):
    values = []
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    if isinstance(tensor, (int, float)):
        return [tensor]
    if isinstance(tensor, list):
        for item in tensor:
            values.extend(flatten_tensor(item))
    elif isinstance(tensor, np.ndarray):
        for item in tensor:
            values.extend(flatten_tensor(item))
    elif isinstance(tensor, np.generic):
        return [tensor.item()]
    return values



def intersectionAndUnion_tensor(imPred, imLab, numClass):
    # for debugging
    # imlab_values_list = flatten_tensor(imLab)
    # unique_imlab_values = list(set(imlab_values_list))
    
    imPred = imPred.to(torch.float32)
    imLab = imLab.to(torch.float32)
    IoU_classes = [0] * numClass
    intersection_classes = [0] * numClass
    union_classes = [0] * numClass
    
    # Compute area intersection:
    for class_idx in range(0, numClass):
        intersection_per_class = torch.sum((imPred == class_idx) & (imLab == class_idx))
        union_per_class = torch.sum(imPred == class_idx) + torch.sum(imLab == class_idx) - intersection_per_class
        intersection_classes[class_idx] += intersection_per_class.item()
        union_classes[class_idx] += union_per_class.item()
        
        IoU_per_class = intersection_per_class/(union_per_class + 1e-10)
        IoU_classes[class_idx] = IoU_per_class.item()
    
    IoU_classes = torch.tensor(IoU_classes)
    intersection_classes = torch.tensor(intersection_classes)    
    union_classes = torch.tensor(union_classes)  

    return intersection_classes, union_classes

# eval train result
def eval_mc_segementation(mask_pred, mask_true, net, device, dice_score, onehot = False):
    ########################################################
    # 1. argmax method, give the label to pred_res directly
    ########################################################
    if onehot == False:
        mask_pred = mask_pred.cpu()
        predicted_mask = torch.argmax(mask_pred, axis=1)
        
        # just for debugging
        # px_values_pred = check_pixel_values(predicted_mask)
        # px_values_true = check_pixel_values(mask_true)
        
        # start eval step
        mask_true = mask_true.to(mask_pred.device)
        predicted_mask = predicted_mask * (mask_true > 0)

        # Compute intersection and union for each class
        intersection, union = intersectionAndUnion_tensor(predicted_mask, mask_true, net.n_classes)
        # IoU_per_class = IoU_classes.to(device)
        union = union.to(device)
        intersection = intersection.to(device)
    
    #########################################
    # 2. one hot method, still waiting to modify
    #########################################
    else:
        # convert to one-hot format
        mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)
        
        mask_pred_one_hot = mask_pred_one_hot * (mask_true_one_hot > 0)
        
        # Compute intersection and union for each class, waiting to modify
        # IoU_classes = intersectionAndUnion_tensor(mask_pred_one_hot, mask_true_one_hot, net.n_classes)
        # IoU_per_class = IoU_classes.to(device)

    IoU_per_class = intersection/ (union + 1e-10)
    mean_iou = IoU_per_class.mean()

    net.train()
    
    
    return IoU_per_class, mean_iou


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    val_loss = 0

    area_intersection = torch.zeros(net.n_classes, device=device)
    area_union = torch.zeros(net.n_classes, device=device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            
            # validation loss
            val_loss += nn.CrossEntropyLoss()(mask_pred, mask_true)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classe]'
                
                # mask_pred = mask_pred.cpu()
                predicted_mask = torch.argmax(mask_pred, axis=1)
                
                # start eval step
                mask_true = mask_true.to(mask_pred.device)
                predicted_mask = predicted_mask * (mask_true > 0)
                
                # Compute intersection and union for each class
                intersection, union = intersectionAndUnion_tensor(predicted_mask, mask_true, net.n_classes)
                union = union.to(device)
                intersection = intersection.to(device)
                area_intersection += intersection
                area_union += union


    IoU_per_class = area_intersection/ (area_union + 1e-10)
    mean_iou = IoU_per_class.mean()
    mean_val_loss = val_loss / max(num_val_batches, 1)

    net.train()
    
    return mean_val_loss, IoU_per_class, mean_iou

