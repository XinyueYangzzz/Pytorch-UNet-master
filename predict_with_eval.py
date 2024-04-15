import argparse
import logging
import os
import glob
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='/home/xinyue/thesis/Pytorch-UNet-master/checkpoints/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default= "/home/xinyue/thesis/Pytorch-UNet-master/data/patches/test/data/" ,metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=7, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    in_files = glob.glob(os.path.join(args.input, '*.tif'))
    out_files = []

    if not args.output:
        for f in in_files:
            basename = os.path.basename(f)
            dirname = os.path.split(os.path.split(f)[0])[0]
            out_files.append(os.path.join(dirname, 'result', f'{os.path.splitext(basename)[0]}_OUT.tif'))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def compute_iou_per_class(predicted_mask, true_mask, num_classes):
    iou_per_class = np.zeros(num_classes)
    for class_id in range(num_classes):
        tp, fp, fn = compute_tp_fp_fn(predicted_mask, true_mask, class_id)
        iou_per_class[class_id] = tp / (tp + fp + fn) if tp + fp + fn != 0 else 0
    
    return iou_per_class


def compute_tp_fp_fn(predicted_mask, true_mask, class_id):
    tp = ((predicted_mask == class_id) & (true_mask == class_id)).sum()
    fp = ((predicted_mask == class_id) & (true_mask != class_id)).sum()
    fn = ((predicted_mask != class_id) & (true_mask == class_id)).sum()
    
    return tp, fp, fn


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    output_dir = os.path.join(os.path.dirname(args.input), 'result')
    os.makedirs(output_dir, exist_ok=True)
    
    true_masks_dir = os.path.join(os.path.dirname(os.path.dirname(args.input)), 'label')
    
    #in_files = args.input
    in_files = glob.glob(os.path.join(args.input, '*.tif'))
    out_files = get_output_filenames(args)
    
    # Initialize lists to store IoU values for each image
    iou_per_image = []
    
    # Initialize lists to store TP, FP, FN for each class
    tp_per_class = np.zeros(args.classes)
    fp_per_class = np.zeros(args.classes)
    fn_per_class = np.zeros(args.classes)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1, 2, 3, 4, 5, 6])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...'.format(filename))
        img = Image.open(filename)
        true_mask_file = os.path.join(true_masks_dir, os.path.basename(filename))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        true_mask = np.array(Image.open(true_mask_file))

        # Compute TP, FP, FN for each class
        for class_id in range(args.classes):
            tp, fp, fn = compute_tp_fp_fn(mask, true_mask, class_id)
            tp_per_class[class_id] += tp
            fp_per_class[class_id] += fp
            fn_per_class[class_id] += fn
        
        # Compute IoU for the current image
        iou = compute_iou_per_class(mask, true_mask, args.classes)
        iou_per_image.append(iou)

        if not args.no_save:
            out_filename = out_files[i]
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
    
    # Compute mean IoU per class
    miou_per_class = np.mean(iou_per_image, axis=0)
    
    # Compute IoU per class
    iou_per_class = np.zeros(args.classes)
    for class_id in range(args.classes):
        iou_per_class[class_id] = tp_per_class[class_id] / (tp_per_class[class_id] + fp_per_class[class_id] + fn_per_class[class_id])

    miou = np.mean(miou_per_class)
    
    logging.info(f'Mean IoU per class: {miou_per_class}')
    #pdb.set_trace()
    logging.info(f'Mean IoU across all classes: {miou}')
    
#python predict_with_eval.py --model "/home/xinyue/thesis/Pytorch-UNet-master/checkpoints/checkpoint_epoch11.pth" --input "/home/xinyue/thesis/Pytorch-UNet-master/data/patches/test/data/"