import io
import torch
import pickle

path = './pretrained_model/coco_resnet_50_map_0_335.pt'
# model = torch.load(path)
# with open(path, 'rb') as f:
#     data = pickle.load(f, encoding='latin1')
with open(path) as f:
    buffer = io.BytesIO(f.read())
model = torch.load(buffer)

print('Load Success')

# print(data)
# 
# model = torch.load(data)
