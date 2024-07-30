# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torchvision
import numpy as np
import torch
from kornia.morphology import opening, erosion
from kornia.filters import gaussian_blur2d
from networks.inpainting_nets import Inpaint_Depth_Net, Inpaint_Color_Net
from core.utils import masked_median_blur
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import repeat

def refine_near_depth_discontinuity(depth, alpha, kernel_size=11):
    '''
    median filtering the depth discontinuity boundary
    '''
    depth = depth * alpha
    depth_median_blurred = masked_median_blur(depth, alpha, kernel_size=kernel_size) * alpha
    alpha_eroded = erosion(alpha, kernel=torch.ones(kernel_size, kernel_size).to(alpha.device))
    depth[alpha_eroded == 0] = depth_median_blurred[alpha_eroded == 0]
    return depth


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):

    image = image * 2 -1
    mask = 1-mask
    masked_image = image * (mask < 0.5)

    batch = {
        # "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

class Diffusion_Inpainter():
    def __init__(self, configs, ckpt, device='cuda'):
        self.device = device
        self.sampler = initialize_model(configs, ckpt)
        # kernels
        self.context_erosion_kernel = torch.ones(10, 10).to(self.device)
        self.alpha_kernel = torch.ones(3, 3).to(self.device)
    
    def inpaint(self, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, 4, h // 8, w // 8)
        start_code = torch.from_numpy(start_code).to(
            device=self.device, dtype=torch.float32)
        with torch.no_grad(), \
            torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt,
                              device=self.device, num_samples=num_samples)
            model = self.sampler.model
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(
                        model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            
            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = self.sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

            return result.float()
    
    def sequential_inpainting(self, rgb, depth, depth_bins, prompt, seed, scale, ddim_steps, dpt_model, w, h):
        '''
        :param rgb: [1, 3, H, W]
        :param depth: [1, 1, H, W]
        :return: rgba_layers: [N, 1, 3, H, W]: the inpainted RGBA layers
                 depth_layers: [N, 1, 1, H, W]: the inpainted depth layers
                 mask_layers:  [N, 1, 1, H, W]: the original alpha layers (before inpainting)
        '''
        num_bins = len(depth_bins) - 1
        nums = 1
        rgba_layers = []
        depth_layers = []
        mask_layers = []
        for i in range(num_bins):
            alpha_i = (depth >= depth_bins[i]) * (depth < depth_bins[i+1])
            alpha_i = alpha_i.float()
            
            if i == 0:
                rgba_i = torch.cat([rgb*alpha_i, alpha_i], dim=1)
                rgba_layers.append(rgba_i)
                depth_i = depth * alpha_i
                depth_layers.append(depth_i)
                mask_layers.append(alpha_i)
                pre_alpha = alpha_i.bool()
                pre_inpainted_depth = depth * alpha_i
            else:
                context = erosion((depth >= depth_bins[i]).float(), self.context_erosion_kernel)
                
                context_rgb = rgb * context
                # inpaint rgb
                inpainted_rgb_i = self.inpaint(context_rgb, context, prompt, seed, scale, ddim_steps, 1, w, h)
                inpainted_a_i = torch.ones((((1,1,inpainted_rgb_i.shape[-2],inpainted_rgb_i.shape[-1]))))
                inpainted_rgba_i = torch.concat((inpainted_rgb_i,inpainted_a_i.cuda()),dim=1)
                # predict depth
                dpt_prediction = dpt_model.forward(inpainted_rgb_i).unsqueeze(0)
                depth_min = dpt_prediction.min()
                depth_max = dpt_prediction.max()
                dpt_prediction = (dpt_prediction - depth_min) / (depth_max - depth_min)
                predict_depth_i = 1. / torch.maximum(dpt_prediction, torch.tensor(1e-2))
                depth_near_mask = (depth < depth_bins[i+1]).float()
                predict_depth_i = predict_depth_i * depth_near_mask
                sort_min = torch.unique(predict_depth_i)[2]
                sort_max = torch.unique(predict_depth_i)[-1]
                if (sort_max-np.percentile(predict_depth_i.cpu().numpy(),99))>10:
                    sort_max=np.percentile(predict_depth_i.cpu().numpy(),99)
                predict_depth_i = torch.clamp((predict_depth_i - sort_min)/(sort_max-sort_min),max=1,min=0)
                inpainted_depth_i = predict_depth_i * (depth_bins[i+1]-depth_bins[i])* 0.35 + depth_bins[i]* 2.0
                
                if i < num_bins - 1:
                    # only keep the content whose depth is smaller than the upper limit of the current layer
                    # otherwise the inpainted content on the far-depth edge will falsely occlude the next layer.
                    inpainted_rgba_i *= depth_near_mask
                    inpainted_depth_i *= depth_near_mask

                # inpainted_alpha_i = inpainted_rgba_i[:, [-1]].bool()
                # mask_wrong_ordering = (inpainted_depth_i <= pre_inpainted_depth) * inpainted_alpha_i
                # inpainted_depth_i[mask_wrong_ordering] = pre_inpainted_depth[mask_wrong_ordering] * 1.05

                rgba_layers.append(inpainted_rgba_i)
                depth_layers.append(inpainted_depth_i)
                mask_layers.append(context * depth_near_mask)   # original mask
                # pre_alpha[inpainted_alpha_i] = True
                # pre_inpainted_depth[inpainted_alpha_i > 0] = inpainted_depth_i[inpainted_alpha_i > 0]

        rgba_layers = torch.stack(rgba_layers)
        depth_layers = torch.stack(depth_layers)
        mask_layers = torch.stack(mask_layers)
        return rgba_layers.float(), depth_layers.float(), mask_layers.float()
    
def refine_near_depth_discontinuity(depth, alpha, kernel_size=11):
    '''
    median filtering the depth discontinuity boundary
    '''
    depth = depth * alpha
    depth_median_blurred = masked_median_blur(depth, alpha, kernel_size=kernel_size) * alpha
    alpha_eroded = erosion(alpha, kernel=torch.ones(kernel_size, kernel_size).to(alpha.device))
    depth[alpha_eroded == 0] = depth_median_blurred[alpha_eroded == 0]
    return depth


def define_inpainting_bbox(alpha, border=40):
    '''
    define the bounding box for inpainting
    :param alpha: alpha map [1, 1, h, w]
    :param border: the minimum distance from a valid pixel to the border of the bbox
    :return: [1, 1, h, w], a 0/1 map that indicates the inpainting region
    '''
    assert alpha.ndim == 4 and alpha.shape[:2] == (1, 1)
    x, y = torch.nonzero(alpha)[:, -2:].T
    h, w = alpha.shape[-2:]
    row_min, row_max = x.min(), x.max()
    col_min, col_max = y.min(), y.max()
    out = torch.zeros_like(alpha)
    x0, x1 = max(row_min - border, 0), min(row_max + border, h - 1)
    y0, y1 = max(col_min - border, 0), min(col_max + border, w - 1)
    out[:, :, x0:x1, y0:y1] = 1
    return out


class Inpainter():
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        print("Loading depth model...")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load('inpainting_ckpts/depth-model.pth', map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        self.depth_feat_model = depth_feat_model.to(device)
        print("Loading rgb model...")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load('inpainting_ckpts/color-model.pth', map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        self.rgb_model = rgb_model.to(device)

        # kernels
        self.context_erosion_kernel = torch.ones(10, 10).to(self.device)
        self.alpha_kernel = torch.ones(3, 3).to(self.device)

    @staticmethod
    def process_depth_for_network(depth, context, log_depth=True):
        if log_depth:
            log_depth = torch.log(depth + 1e-8) * context
            mean_depth = torch.mean(log_depth[context > 0])
            zero_mean_depth = (log_depth - mean_depth) * context
        else:
            zero_mean_depth = depth
            mean_depth = 0
        return zero_mean_depth, mean_depth

    @staticmethod
    def deprocess_depth(zero_mean_depth, mean_depth, log_depth=True):
        if log_depth:
            depth = torch.exp(zero_mean_depth + mean_depth)
        else:
            depth = zero_mean_depth
        return depth

    def inpaint_rgb(self, holes, context, context_rgb, edge):
        # inpaint rgb
        with torch.no_grad():
            inpainted_rgb = self.rgb_model.forward_3P(holes, context, context_rgb, edge,
                                                      unit_length=128, cuda=self.device)
        inpainted_rgb = inpainted_rgb.detach() * holes + context_rgb
        inpainted_a = holes + context
        inpainted_a = opening(inpainted_a, self.alpha_kernel)
        inpainted_rgba = torch.cat([inpainted_rgb, inpainted_a], dim=1)
        return inpainted_rgba

    def inpaint_depth(self, depth, holes, context, edge, depth_range):
        zero_mean_depth, mean_depth = self.process_depth_for_network(depth, context)
        with torch.no_grad():
            inpainted_depth = self.depth_feat_model.forward_3P(holes, context, zero_mean_depth, edge,
                                                               unit_length=128, cuda=self.device)
        inpainted_depth = self.deprocess_depth(inpainted_depth.detach(), mean_depth)
        inpainted_depth[context > 0.5] = depth[context > 0.5]
        inpainted_depth = gaussian_blur2d(inpainted_depth, (3, 3), (1.5, 1.5))
        inpainted_depth[context > 0.5] = depth[context > 0.5]
        # if the inpainted depth in the background is smaller that the foreground depth,
        # then the inpainted content will mistakenly occlude the foreground.
        # Clipping the inpainted depth in this situation.
        mask_wrong_depth_ordering = inpainted_depth < depth
        inpainted_depth[mask_wrong_depth_ordering] = depth[mask_wrong_depth_ordering] * 1.01
        inpainted_depth = torch.clamp(inpainted_depth, min=min(depth_range)*0.9)
        return inpainted_depth

    def sequential_inpainting(self, rgb, depth, depth_bins):
        '''
        :param rgb: [1, 3, H, W]
        :param depth: [1, 1, H, W]
        :return: rgba_layers: [N, 1, 3, H, W]: the inpainted RGBA layers
                 depth_layers: [N, 1, 1, H, W]: the inpainted depth layers
                 mask_layers:  [N, 1, 1, H, W]: the original alpha layers (before inpainting)
        '''

        num_bins = len(depth_bins) - 1

        rgba_layers = []
        depth_layers = []
        mask_layers = []

        for i in range(num_bins):
            alpha_i = (depth >= depth_bins[i]) * (depth < depth_bins[i+1])
            alpha_i = alpha_i.float()

            if i == 0:
                rgba_i = torch.cat([rgb*alpha_i, alpha_i], dim=1)
                rgba_layers.append(rgba_i)
                depth_i = refine_near_depth_discontinuity(depth, alpha_i)
                depth_layers.append(depth_i)
                mask_layers.append(alpha_i)
                pre_alpha = alpha_i.bool()
                pre_inpainted_depth = depth * alpha_i
            else:
                alpha_i_eroded = erosion(alpha_i, self.context_erosion_kernel)
                if alpha_i_eroded.sum() < 10:
                    continue
                context = erosion((depth >= depth_bins[i]).float(), self.context_erosion_kernel)
                holes = 1. - context
                bbox = define_inpainting_bbox(context, border=40)
                holes *= bbox
                edge = torch.zeros_like(holes)
                context_rgb = rgb * context
                # inpaint depth
                inpainted_depth_i = self.inpaint_depth(depth, holes, context, edge, (depth_bins[i], depth_bins[i+1]))
                depth_near_mask = (inpainted_depth_i < depth_bins[i+1]).float()
                # inpaint rgb
                inpainted_rgba_i = self.inpaint_rgb(holes, context, context_rgb, edge)

                if i < num_bins - 1:
                    # only keep the content whose depth is smaller than the upper limit of the current layer
                    # otherwise the inpainted content on the far-depth edge will falsely occlude the next layer.
                    inpainted_rgba_i *= depth_near_mask
                    inpainted_depth_i = refine_near_depth_discontinuity(inpainted_depth_i, inpainted_rgba_i[:, [-1]])

                inpainted_alpha_i = inpainted_rgba_i[:, [-1]].bool()
                mask_wrong_ordering = (inpainted_depth_i <= pre_inpainted_depth) * inpainted_alpha_i
                inpainted_depth_i[mask_wrong_ordering] = pre_inpainted_depth[mask_wrong_ordering] * 1.05

                rgba_layers.append(inpainted_rgba_i)
                depth_layers.append(inpainted_depth_i)
                mask_layers.append(context * depth_near_mask)   # original mask

                pre_alpha[inpainted_alpha_i] = True
                pre_inpainted_depth[inpainted_alpha_i > 0] = inpainted_depth_i[inpainted_alpha_i > 0]

        rgba_layers = torch.stack(rgba_layers)
        depth_layers = torch.stack(depth_layers)
        mask_layers = torch.stack(mask_layers)

        return rgba_layers, depth_layers, mask_layers