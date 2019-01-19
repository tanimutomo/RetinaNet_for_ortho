import torch

try:
    model = torch.load('./pretrained_model/coco_resnet_50_map_0_335.pt')
    print('Load Success')
except:
    print('Load Fail')
