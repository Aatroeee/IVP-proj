from transformers import pipeline
from PIL import Image
import argparse
import IPython
import sys

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    return parser.parse_args()

def estimate_depth(image_path):
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",device="cuda")
    image = Image.open(image_path)
    depth = pipe(image)["depth"]
    return depth

def align_depth(depth_data, estimated_depth, input_mask):
    # depth_data: (b, qo, ho, wo)
    # estimated_depth: (b, qo, ho, wo)
    # input_mask: (b, qo, ho, wo)
    #NOTE: only calculate scale&shift on non-masked
    # Get only the masked depth values on (b,q)
    b, qo, ho, wo = depth_data.shape
    depth_data_reshaped = depth_data.reshape(b*qo, ho*wo)  # (b*qo, ho*wo)
    estimated_depth_reshaped = estimated_depth.reshape(b*qo, ho*wo)  # (b*qo, ho*wo) 
    input_mask_reshaped = input_mask.reshape(b*qo, ho*wo)  # (b*qo, ho*wo)
    
    valid_mask = input_mask_reshaped > 0.5  # (b*qo, ho*wo)
    
    img_n = b * qo
    valid_depth = [depth_data_reshaped[i][valid_mask[i]] for i in range(img_n)]
    valid_estimated = [estimated_depth_reshaped[i][valid_mask[i]] for i in range(img_n)]
    
    median_valid_depth = [torch.median(valid_depth[i]).unsqueeze(-1) for i in range(img_n)]
    median_valid_estimated = [torch.median(valid_estimated[i]).unsqueeze(-1) for i in range(img_n)]
    # IPython.embed()
    # sys.exit()
    estimated_depth_stacked = torch.stack(median_valid_estimated, dim=0)  # (b*qo, 1)
    valid_depth_disp = [valid_depth[i] - median_valid_depth[i] + 1e-8 for i in range(img_n)]
    valid_estimated_disp = [valid_estimated[i] - median_valid_estimated[i] + 1e-8 for i in range(img_n)]
    
    scale = [torch.median(valid_depth_disp[i] / valid_estimated_disp[i], dim=-1)[0].unsqueeze(-1) for i in range(img_n)]
    shift = [torch.median(valid_depth_disp[i] - scale[i] * valid_estimated_disp[i], dim=-1)[0].unsqueeze(-1) for i in range(img_n)]
    
    scale = torch.stack(scale, dim=0)  # (b*qo, 1)
    shift = torch.stack(shift, dim=0)  # (b*qo, 1)
    
    full_estimated_depth_disp = estimated_depth_reshaped - estimated_depth_stacked + 1e-8
    aligned_depth = scale * full_estimated_depth_disp + shift
    aligned_depth = aligned_depth.reshape(b, qo, ho, wo)
    return aligned_depth

if __name__ == "__main__":
    args = arg_parser()
    depth = estimate_depth(args.img)

    depth.save("res/depth.png")
    print(depth)