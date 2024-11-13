import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def convert_image_to_tensor(img):
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)
    return img_pt


def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)
    return Image.fromarray(result_rgb_np)

class PixelEffectModule(nn.Module):
    def __init__(self):
        super(PixelEffectModule, self).__init__()

    def create_mask_by_idx(self, idx_z, max_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z])
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def forward(self, rgb, param_num_bins, param_kernel_size, param_pixel_size):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256. * param_num_bins
        intensity_idx = intensity_idx.long()

        intensity = self.create_mask_by_idx(intensity_idx, max_z=param_num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0)

        r, g, b = r * intensity, g * intensity, b * intensity

        kernel_conv = torch.ones([param_num_bins, 1, param_kernel_size, param_kernel_size])
        r = F.conv2d(input=r, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        g = F.conv2d(input=g, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        b = F.conv2d(input=b, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        intensity = F.conv2d(input=intensity, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins,
                             bias=None)[0, :, :, :]
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)


        r = torch.permute(r, dims=[1, 2, 0])
        g = torch.permute(g, dims=[1, 2, 0])
        b = torch.permute(b, dims=[1, 2, 0])

        r = self.select_by_idx(r, intensity_argmax)
        g = self.select_by_idx(g, intensity_argmax)
        b = self.select_by_idx(b, intensity_argmax)

        r = r / intensity_max
        g = g / intensity_max
        b = b / intensity_max

        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb_scale = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        return result_rgb,result_rgb_scale

class Photo2PixelModel(nn.Module):
    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()

    def forward(self, rgb,
                param_kernel_size=10,
                param_pixel_size=16):
        rgb,rgb_scale = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)
        return rgb,rgb_scale
    

def resize_image(image, max_size, is_pixel=False):
    # Image.LANCZOS,Image.NEAREST
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    if is_pixel:
        sample_type = Image.NEAREST
    else:
        sample_type = Image.LANCZOS
    resized_image = image.resize((new_width, new_height), sample_type)
    
    return resized_image

def convert_photo_to_pixel(img,abstraction,pixel_size):
    pixel_tile_size = 16
    preview_size = 256
    img = resize_image(img,pixel_size*pixel_tile_size)
    img_tensor = convert_image_to_tensor(img)
    model = Photo2PixelModel()
    model.eval()
    
    with torch.no_grad():
        rgb,rgb_scale = model(img_tensor,param_kernel_size = abstraction,param_pixel_size = pixel_tile_size)
    
    img_output = convert_tensor_to_image(rgb)
    img_preview = convert_tensor_to_image(rgb_scale)
    img_preview = resize_image(img_preview,preview_size,is_pixel=True)
    return img_output,img_preview



def convert(input_path,output_path,abstraction,pixel_size):
    img_input = Image.open(input_path)
    img_output,img_preview = convert_photo_to_pixel(img_input,abstraction,pixel_size)
    img_output.save(output_path)
    img_preview.save(output_path.replace('.png','_preview.png'))


if __name__ == '__main__':
    input_path = '/home/badger/code/badgertools/photo2pixel/test.png'
    output_path = '/home/badger/code/badgertools/photo2pixel/test_.png'
    convert(input_path,output_path,15,64)
