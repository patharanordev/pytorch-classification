#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image

from reshape import reshape_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#
# parse command-line arguments
#
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--image', default='', type=str, metavar='PATH',
                    help='path to target image (default: none)')
parser.add_argument('--classes', default='', type=str,
                    help='Class or category names that separate by comma(,).')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

#
# initiate worker threads (if using distributed multi-GPU)
#
def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

#
# worker thread (per-GPU)
#
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    classes = args.classes.split(',')

    num_classes = len(classes)
    print('=> dataset classes:  ' + str(num_classes) + ' ' + str(classes))

    # create or load the model if using pre-trained (the default)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](weights='DEFAULT')
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # reshape the model for the number of classes in the dataset
    model = reshape_model(model, args.arch, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            start_time = time.time()
            checkpoint = torch.load(args.resume, weights_only=True)
            end_time = time.time()
            print(f"=> loading time without mmap={end_time - start_time}")

            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            
            image = Image.open(args.image).convert('RGB')

            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            
            img_preprocessed = preprocess(image)
            batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)

            start_time = time.time()
            with torch.no_grad():
                preds = model(batch_img_tensor.to(args.gpu))
            end_time = time.time()
            spend_time = end_time - start_time

            outputs = torch.nn.functional.softmax(preds, dim=1)
            max_elements, max_idxs = torch.max(outputs[0], dim=0)
            print("=> - max_elements: ", max_elements)
            print("=> - max_idxs: ", max_idxs)

            print(f"=> The predicted result (spend {spend_time} seconds): ", outputs)
            print(f"=> The predicted category is: {classes[max_idxs]}")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

if __name__ == '__main__':
    main()
