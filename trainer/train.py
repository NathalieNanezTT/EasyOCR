import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number = 2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = torch.nn.DataParallel(model).to(device) 
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    print("Model:")
    print(model)
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    while(True):
        # train part
        optimizer.zero_grad(set_to_none=True)
        
        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
            optimizer.step()
        loss_avg.add(cost)

        # validation part
        if (i % opt.valInterval == 0) and (i!=0):
            print('training time: ', time.time()-t1)
            t1=time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                #show_number = min(show_number, len(labels))
                
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time()-t1)
                t1=time.time()
        # save model per 1e+4 iter.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1




import yaml
import argparse

def get_opt():
    # Load the configuration from the YAML file
    with open('config_files/en_filtered_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize argparse
    parser = argparse.ArgumentParser()

    # Add arguments from the YAML config
    parser.add_argument('--train_data', type=str, default=config['train_data'], help='Path to training data')
    parser.add_argument('--valid_data', type=str, default=config['valid_data'], help='Path to validation data')
    parser.add_argument('--select_data', type=str, default=config['select_data'], help='Sub-directory within the training data')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'], help='Batch size')
    parser.add_argument('--num_iter', type=int, default=config['num_iter'], help='Number of iterations to train')
    parser.add_argument('--workers', type=int, default=config['workers'], help='Number of worker threads for data loading')
    parser.add_argument('--valInterval', type=int, default=config['valInterval'], help='Validation interval')
    parser.add_argument('--saved_model', type=str, default=config['saved_model'], help='Path to pre-trained model')
    parser.add_argument('--FT', type=bool, default=config['FT'], help='Whether to perform fine-tuning')
    parser.add_argument('--optim', type=bool, default=config['optim'], help='Optimizer setting (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=config['lr'], help='Learning rate')
    parser.add_argument('--beta1', type=float, default=config['beta1'], help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--rho', type=float, default=config['rho'], help='Rho parameter for Adadelta optimizer')
    parser.add_argument('--eps', type=float, default=config['eps'], help='Epsilon for optimizer')
    parser.add_argument('--grad_clip', type=float, default=config['grad_clip'], help='Gradient clipping value')
    parser.add_argument('--contrast_adjust', type=bool, default=config['contrast_adjust'], help='Whether to apply contrast adjustment')
    parser.add_argument('--data_filtering_off', type=bool, default=config['data_filtering_off'], help='Disable data filtering')
    parser.add_argument('--batch_ratio', type=str, default=config['batch_ratio'], help='Batch ratio for dataset')
    parser.add_argument('--total_data_usage_ratio', type=float, default=config['total_data_usage_ratio'], help='Total data usage ratio (0 to 1)')
    parser.add_argument('--batch_max_length', type=int, default=config['batch_max_length'], help='Maximum length of text labels')
    parser.add_argument('--imgH', type=int, default=config['imgH'], help='Image height')
    parser.add_argument('--imgW', type=int, default=config['imgW'], help='Image width')
    parser.add_argument('--rgb', type=bool, default=config['rgb'], help='Whether to use RGB images')
    parser.add_argument('--sensitive', type=bool, default=config['sensitive'], help='Whether to apply case sensitivity')
    parser.add_argument('--PAD', type=bool, default=config['PAD'], help='Whether to pad images')
    
    # Model architecture parameters
    parser.add_argument('--Transformation', type=str, default=config['Transformation'], choices=['None', 'TPS', 'Warp'], help='Transformation method')
    parser.add_argument('--FeatureExtraction', type=str, default=config['FeatureExtraction'], choices=['VGG', 'ResNet'], help='Feature extraction method')
    parser.add_argument('--SequenceModeling', type=str, default=config['SequenceModeling'], choices=['None', 'BiLSTM', 'Transformer'], help='Sequence modeling method')
    parser.add_argument('--Prediction', type=str, default=config['Prediction'], choices=['CTC', 'Attn'], help='Prediction method')
    parser.add_argument('--num_fiducial', type=int, default=config['num_fiducial'], help='Number of fiducial points')
    parser.add_argument('--input_channel', type=int, default=config['input_channel'], help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--output_channel', type=int, default=config['output_channel'], help='Output channels for feature extraction')
    parser.add_argument('--hidden_size', type=int, default=config['hidden_size'], help='Hidden size of the model')
    parser.add_argument('--decode', type=str, default=config['decode'], choices=['greedy', 'beamsearch'], help='Decoding method')
    parser.add_argument('--new_prediction', type=bool, default=config['new_prediction'], help='Whether to use a new prediction module')
    parser.add_argument('--freeze_FeatureFxtraction', type=bool, default=config['freeze_FeatureFxtraction'], help='Freeze the feature extraction layers')
    parser.add_argument('--freeze_SequenceModeling', type=bool, default=config['freeze_SequenceModeling'], help='Freeze the sequence modeling layers')

    # ** Experiment name **
    parser.add_argument('--experiment_name', type=str, default=config['experiment_name'], help='Name of the experiment for saving model')

    # ** lang_char as character **
    parser.add_argument('--character', type=str, default=config['lang_char'], help='Characters to recognize')

    # Return parsed arguments
    return parser.parse_args()

if __name__ == "__main__":
    opt = get_opt()  # Get the parsed arguments
    print(opt)

    train(opt)  # Call the train function with the parsed options