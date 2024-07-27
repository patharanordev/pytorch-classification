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
parser.add_argument('--images', default='', type=str, metavar='PATH',
                    help='path to target images (default: none)')
parser.add_argument('--classes', default='', type=str,
                    help='Class or category names that separate by comma(,).')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

class ImageClassification(object):
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device('cuda')
        self.classes = []
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        
    def set_classes(self, classes_str:str):
        self.classes = classes_str.split(',')
            
    def load_pretrained_model(self, arch, model_fpath:str):
        print('=> using pre-trained model "{}"'.format(arch))
        self.model = models.__dict__[arch](weights='DEFAULT')

        num_classes = len(self.classes)
        print(f'=> classes ({str(num_classes)}): {str(self.classes)}')
        
        # reshape the model for the number of classes in the dataset
        self.model = reshape_model(self.model, arch, num_classes)
        self.model.to(self.device)

        start_time = time.time()
        checkpoint = torch.load(model_fpath, weights_only=True)
        end_time = time.time()
        print(f"=> loading time without mmap={end_time - start_time}")

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, image:Image):
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        img_preprocessed = preprocess(image)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)

        start_time = time.time()
        with torch.no_grad():
            preds = self.model(batch_img_tensor.to(self.device))
        end_time = time.time()
        spend_time = end_time - start_time

        outputs = torch.nn.functional.softmax(preds, dim=1)
        max_elements, max_idxs = torch.max(outputs[0], dim=0)
        print("=> - max_elements: ", max_elements)
        print("=> - max_idxs: ", max_idxs)
        print(f"=> The predicted result (spend {spend_time} seconds): ", outputs)

        return self.classes[max_idxs]

def main():
    args = parser.parse_args()
    
    img_cls = ImageClassification()
    img_cls.set_classes(args.classes)

    # optionally resume from a checkpoint
    if args.resume and args.images:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            img_cls.load_pretrained_model(
                arch=args.arch, 
                model_fpath=args.resume)
            
            images = []
            image_fpaths = args.images.split(',')
            for image_fpath in image_fpaths:
                if os.path.isfile(image_fpath):
                    images.append(Image.open(image_fpath).convert('RGB'))

            for image in images:
                label = img_cls.predict(image)
                print(f"=> The predicted category is: {label}")

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

if __name__ == '__main__':
    main()
