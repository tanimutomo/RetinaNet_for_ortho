import torch

path = './pretrained_model/coco_resnet_50_map_0_335.pt'
# model = torch.load(path)
with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print('Load Success')
