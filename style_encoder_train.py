import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import torch.optim as optim
from utils.auxilary_functions import affine_transformation
from feature_extractor import ImageEncoder
from style_encoder_modules.data import IAMDataset_style
from style_encoder_modules.training import train_mixed, train_classification, train_triplet as train


def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Train Style Encoder')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--dataset', type=str, default='iam', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=320, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, required=False, help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, help='use of feature extractor or not')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--save_path', type=str, default='./style_models', help='path to save models')
    parser.add_argument('--mode', type=str, default='mixed', help='mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    #========= Data augmentation and normalization for training =====#
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    
    if args.dataset == 'iam':
    
        myDataset = IAMDataset_style
        # dataset_folder = '/usr/share/datasets_ianos'
        dataset_folder = '/path/to/iam_data/'
        aug_transforms = [lambda x: affine_transformation(x, s=.1)]
        
        train_transform = transforms.Compose([
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        
        val_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        
        #train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        # full_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=train_transform)
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=train_transform)
        
        #print('len train data', len(train_data))
        #split with torch.utils.data.Subset into train and val
        # validation_size = int(0.2 * len(full_data))

        # Calculate the size of the training set
        # train_size = len(full_data) - validation_size

        # Use random_split to split the dataset into train and validation sets
        # train_data, val_data = random_split(full_data, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
        # print('len train data', len(train_data))
        # print('len val data', len(val_data))
        
        num_workers = min(2, os.cpu_count() or 1)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
        
        # val_data/val_loader in this training script.
        # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        val_loader = None
        print('len train data', len(train_data))
        print('Validation disabled in training script (run separately).')
        
        # max_label = max(sample[2] for sample in full_data.data) if len(full_data.data) > 0 else -1
        # style_classes = max(full_data.wclasses, int(max_label) + 1)
        max_label = max(sample[2] for sample in train_data.data) if len(train_data.data) > 0 else -1
        style_classes = max(train_data.wclasses, int(max_label) + 1)
    
    else:
        print('You need to add your own dataset and define the number of style classes!!!')
    
    
    
    
    if args.model == 'mobilenetv2_100':
        print('Using mobilenetv2_100')
        model = ImageEncoder(model_name='mobilenetv2_100', num_classes=style_classes, pretrained=True, trainable=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if args.pretrained == True:
            
            state_dict = torch.load(PATH, map_location=args.device)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            #print(model)
            print('Pretrained mobilenetv2_100 model loaded')
        
        
    if args.model == 'resnet18':
        print('Using resnet18')
        model = ImageEncoder(model_name=args.model, num_classes=style_classes, pretrained=True, trainable=True)
        print('Model loaded')
        #change layer to have 1 channel instead of 3
        #model.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if args.pretrained == True:
            PATH = ''
            
            state_dict = torch.load(PATH, map_location=args.device)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            
    
    
    model = model.to(device)
    #print(model)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    #THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == 'mixed':
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2) 
        print('Using both classification and metric learning training')
        train_mixed(model, train_loader, val_loader, criterion_triplet, None, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
    if args.mode == 'triplet':
        train(model, train_loader, val_loader, criterion, optimizer_ft, lr_scheduler, device, args)
        print('finished training')
    
    
    elif args.mode == 'classification':
        
        train_classification(model, train_loader, val_loader, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
if __name__ == '__main__':
    main()
