import torch
import numpy as np


def adjust_for_ortho(boxes, position, div_num):
    for idx, box in enumerate(boxes):
        tl_x = box[0]
        tl_y = box[1]
        br_x = box[2]
        br_y = box[3]
        # start position from 0 not 1
        adj_x = (position[1] - 1 - 11) * 600
        adj_y = (position[0] - 1 - 8) * 600

        out_box = torch.tensor([
            tl_x + adj_x,
            tl_y + adj_y,
            br_x + adj_x,
            br_y + adj_y
            ]).unsqueeze(0)

        if idx == 0:
            out_boxes = out_box
        else:
            out_boxes = torch.cat(
                (out_boxes, out_box), 0)
        
    return out_boxes
    

# this is tmporal implementation term presentation
# so, this function is stricted 3 x 3 ortho image
def unite_images(images, idxs, positions, div_nums):
    # all element in div_nums(list) is same.
    div_num = div_nums[0]
    print(images[0])
    
