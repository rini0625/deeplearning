## 210106, 학습 완료 후 추론을 위한 코드
# 저장된 best model 경로만 설정
# 수행 명령어
# >> python inference.py
# 출력: test 데이터셋에 대한 loss와 정확도, f1 score
# ex) loss|acc|f1 : 0.011 | 99.91 | 99.78

weights_path = 'output/model_2_100.00_100.00.pt'

## load library
import numpy as np
import json
# from PIL import Image
# import PIL.Image as pilimg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from sklearn.metrics import f1_score
from tqdm import tqdm


## parameter
is_Test = False
# is_Test = True
num_epochs = 25
batch_size  = 128

data_path = 'data_220104_split'
save_path='output_2'



## prepare data
## make dataset
from torchvision import transforms, datasets
from torch.utils.data import Subset, dataloader
# class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
# data_train_path = os.path.join(data_path, 'train')
# data_valid_path = os.path.join(data_path, 'valid')
data_test_path  = os.path.join(data_path, 'test')

# 이미지 tensor형태로 변환
transform_function = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력사이즈로 resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = {}
# dataset['train'] = datasets.ImageFolder(data_train_path, 
#                                         transform_function)
# dataset['valid'] = datasets.ImageFolder(data_valid_path,
#                                         transform_function)
dataset['test'] = datasets.ImageFolder(data_test_path,
                                        transform_function)
# print('data proportion(train:valid:test) = %s : %s : %s'%(len(dataset['train']), len(dataset['valid']), len(dataset['test'])))



## data loader 선언
dataloaders, batch_num = {}, {}
# dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
#                                               batch_size=batch_size, shuffle=True,
#                                               num_workers=4)
# dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
#                                               batch_size=batch_size, shuffle=False,
#                                               num_workers=4)
dataloaders['test']  = torch.utils.data.DataLoader(dataset['test'],
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4)
# batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
# print('batch_size : %d,  number of batch(tvt) : %d / %d / %d' % (batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))



## load model for test
def load_model_for_test(weights_path):
    
    # load best model from weight
    # weights_path = 'output_crop/model_4_100.00_100.00.pt'


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load model
    from efficientnet_pytorch import EfficientNet
    model_name = 'efficientnet-b0'  # b5
    num_classes = 2  # 장싱, 비정상
    freeze_extractor = True  # FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
    use_multi_gpu = True

    model_load = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)  # load weight
    model_load.load_state_dict(state_dict, strict=False)  # insert weight to model structure

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)            
    print('학습 parameters 개수 : %d'%(count_parameters(model_load)))

    # multi gpu(2개 이상)를 사용하는 경우
    if use_multi_gpu:
        num_gpu = torch.cuda.device_count()
        if (device.type=='cuda') and (num_gpu > 1):
            print('use multi gpu : %d' % (num_gpu))
            model_load = nn.DataParallel(model_load, device_ids=list(range(num_gpu)))

    model_load = model_load.to(device)
    model_load.eval()

    # define optimizer, criterion
    criterion = nn.CrossEntropyLoss()  # 분류이므로 cross entrophy 사용    
    
    return model_load, criterion, device

model_load, criterion, device = load_model_for_test(weights_path)



# get_test_metric
def get_test_metric(model, phase = 'test', num_images=4, device='cuda', is_Test=False):

    class_names = ['anomal', 'normal']
    was_training = model.training
    model.eval()

    running_loss, running_corrects, num_cnt = 0.0, 0, 0
    pred_list, label_list, file_path_list = [], [], [] 


    dataloader = dataloaders[phase]
    allFiles, _ = map(list, zip(*dataloader.dataset.samples))

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[phase]):

            if is_Test:
                if idx > 2: break

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

            pred_list  += preds.data.cpu().numpy().tolist()
            label_list += labels.data.cpu().numpy().tolist()
            file_path_list += allFiles[idx*batch_size : idx * batch_size+inputs.size(0)]

        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt
        test_f1   = float(f1_score(label_list, pred_list, average='macro'))  # micro    
        print('test done : loss|acc|f1 : %.3f | %.2f | %.2f ' % (test_loss, test_acc*100, test_f1*100))

    return label_list, pred_list, file_path_list


## TEST!
label_list, pred_list, file_path_list = get_test_metric(model=model_load, num_images=4, device=device)




## 220119, metric 출력때문에 추가 ############################################################

## get result
# true/pred  amomal  normal
# anomal     TP(0->0)      FN(0->1)
# normal     FP(1->0)      TN(1->1)
Plot_File_Path = True  # 파일이름을 출력할것인가 말것인가
# Plot_File_Path = False  # 파일이름을 출력할것인가 말것인가

from sklearn.metrics import confusion_matrix
def get_result(label_list, pred_list, file_path_list=False, verbose=False):
    
    # ex) result = get_result(label_list, pred_list)

    CONFUSION = confusion_matrix(label_list, pred_list)

    TP = CONFUSION[0][0]  # 0->0
    
    
    if len(CONFUSION) > 1:
        TN = CONFUSION[1][1]  # 1->1
        FN = CONFUSION[0][1]  # 0->1
        FP = CONFUSION[1][0]  # 1->0
    else:
        TN = 0
        FN = 0
        FP = 0

    ACCURACY  = (TP + TN) / (TP + TN + FP + FN)
    PRECISION = (TP) / (TP + FP)
    RECALL    = (TP) / (TP + FN)
    F1        = (2 * PRECISION * RECALL) / (PRECISION + RECALL)

    result = {}
    result['TP'] = TP
    result['TN'] = TN
    result['FN'] = FN
    result['FP'] = FP
    
    result['ACCURACY'] = ACCURACY
    result['PRECISION'] = PRECISION
    result['RECALL'] = RECALL
    result['F1'] = F1
    if file_path_list:
        result['FILE'] = file_path_list

    if verbose:
        print('confusion matrix\n', CONFUSION)
        print('TP | TN | FN | FP : %.3f | %.3f | %.3f | %.3f' % (TP, TN, FN, FP))
        print('Accuracy : %.3f\nPrecision: %.3f\nRecall   : %.3f\nF1       : %.3f'%(ACCURACY, PRECISION, RECALL, F1))
    
    return result

# result = get_result(label_list, pred_list)

print('Label | Pred  | TP    | TN    | FP    | FN    | Accuracy | Precision | Recall | F1-Score')

if Plot_File_Path:
    header = ['File', 'Label', 'Predict', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
else:
    header = ['Label', 'Predict', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

list_print_txt, list_print_csv = [], []
for idx in range(len(label_list)):
    if Plot_File_Path:
        result = get_result(label_list[:idx+1], pred_list[:idx+1], file_path_list[:idx+1])
        print_txt = '%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f' % (result['FILE'][idx], label_list[idx], pred_list[idx], result['TP'], result['TN'], result['FP'], result['FN'], result['ACCURACY'], result['PRECISION'], result['RECALL'], result['F1'])
        print_csv = [result['FILE'][idx], label_list[idx], pred_list[idx], result['TP'], result['TN'], result['FP'], result['FN'], result['ACCURACY'], result['PRECISION'], result['RECALL'], result['F1']]
    else:
        result = get_result(label_list[:idx+1], pred_list[:idx+1])
        print_txt = '%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f' % (label_list[idx], pred_list[idx], result['TP'], result['TN'], result['FP'], result['FN'], result['ACCURACY'], result['PRECISION'], result['RECALL'], result['F1'])
        print_csv = [label_list[idx], pred_list[idx], result['TP'], result['TN'], result['FP'], result['FN'], result['ACCURACY'], result['PRECISION'], result['RECALL'], result['F1']]
    
    list_print_txt.append(print_txt)
    list_print_csv.append(print_csv)    
for tmp in list_print_txt: print(tmp)

## save csv
import csv
with open('result.csv','w', newline='') as f_csv:
    wr = csv.writer(f_csv)
    wr.writerow(header)
    wr.writerows(list_print_csv)


# print final result
result = get_result(label_list, pred_list)
print('              Accuracy | Precision | Recall  | F1-Score')
print_txt = 'Final result: %.3f    | %.3f     | %.3f   | %.3f' % (result['ACCURACY'], result['PRECISION'], result['RECALL'], result['F1'])
print(print_txt)