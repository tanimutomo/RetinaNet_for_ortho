import torch
from torchvision import datasets, models, transforms
import new_model
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


dataset_train = CocoDataset("./data", set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
retinanet = new_model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)

retinanet.load_state_dict(torch.load("./saved_models/model_final_0.pth", map_location='cuda:0'))

print(retinanet)
print(retinanet.anchors)
