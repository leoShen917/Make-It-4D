from core.inpainter import Inpainter
import config
import random
import numpy as np
import torch.nn.functional as F
import imageio
import cv2
import os 
import torch
from core.inpainter import Inpainter,Diffusion_Inpainter
from core.renderer import ImgRenderer
from model import SpaceTimeModel
from third_party.DPT.dpt.models import DPTDepthModel
import torchvision
import torch.nn.functional as F
from third_party.DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
EPSILON = 1e-8  # small number to avoid numerical issues

def normalize(x):
      return x / np.linalg.norm(x)
  
def pose_from_look_direction_np(camera_pos, camera_dir, camera_up):
  """Computes poses given camera parameters, in numpy.

  Args:
    camera_pos: camera position.
    camera_dir: camera looking direction
    camera_up: camera up vector.

  Returns:
    (R, t) rotation and translation of given camera parameters
  """

  camera_right = normalize(np.cross(camera_up, camera_dir))
  rot = np.zeros((4, 4))
  rot[0, 0:3] = normalize(camera_right)
  rot[1, 0:3] = normalize(np.cross(camera_dir, camera_right))
  rot[2, 0:3] = normalize(camera_dir)
  rot[3, 3] = 1
  trans_matrix = np.array([[1.0, 0.0, 0.0, -camera_pos[0]],
                           [0.0, 1.0, 0.0, -camera_pos[1]],
                           [0.0, 0.0, 1.0, -camera_pos[2]],
                           [0.0, 0.0, 0.0, 1.0]])
  tmp = rot @ trans_matrix
  return tmp[:3, :3], tmp[:3, 3]

def pose_from_look_direction(camera_pos, camera_dir, down_direction):
  """Computes poses given camera parameters, in Pytorch.

  Args:
    camera_pos: camera position.
    camera_dir: camera looking direction
    down_direction: camera down vector.

  Returns:
    (R, t) rotation and translation of given camera parameters
  """

  camera_right = F.normalize(
      torch.cross(down_direction, camera_dir, dim=1), dim=-1)
  rot = torch.eye(4).unsqueeze(0).repeat(camera_pos.shape[0], 1, 1)
  rot[:, 0, 0:3] = (camera_right)
  rot[:, 1, 0:3] = F.normalize(
      torch.cross(camera_dir, camera_right, dim=1), dim=-1)
  rot[:, 2, 0:3] = F.normalize(camera_dir, dim=-1)

  # this is equivalent to inverse matrix
  trans_matrix = torch.eye(4).unsqueeze(0).repeat(camera_pos.shape[0], 1, 1)
  trans_matrix[:, 0:3, 3] = -camera_pos[:, 0:3]

  pose_inv = rot @ trans_matrix
  return pose_inv[:, :3, :3], pose_inv[:, :3, 3]


def skyline_balance(disparity,
                    horizon,
                    sky_threshold,
                    near_fraction):
  """Computes movement parameters from a disparity image.

  Args:
    disparity: current disparity image
    horizon: how far down the image the horizon should ideally be.
    sky_threshold: target sky percentage
    near_fraction: target near content percentage

  Returns:
    (x, y, h) where x and y are where in the image we want to be looking
    and h is how much we want to move upwards.
  """
  sky = torch.clamp(20.0 * (sky_threshold - disparity), 0.0, 1.0)
  # How much of the image is sky?
  sky_fraction = torch.mean(sky, dim=[1, 2])
  y = torch.clamp(0.5 + sky_fraction - horizon, 0.0, 1.0)

  # The balance of sky in the left and right half of the image.
  w2 = disparity.shape[-1] // 2
  sky_left = torch.mean(sky[:, :, :w2], dim=[1, 2])
  sky_right = torch.mean(sky[:, :, w2:], dim=[1, 2])
  # Turn away from mountain:
  x = (sky_right + EPSILON) / (sky_left + sky_right + 2 * EPSILON)

  # Now we try to measure how "near the ground" we are, by looking at how
  # much of the image has disparity > 0.4 (ramping to max at 0.5)
  ground_t = 0.4
  ground = torch.clamp(10.0 * (disparity - ground_t), 0.0, 1.0)
  ground_fraction = torch.mean(ground, dim=[1, 2])
  h = horizon + (near_fraction - ground_fraction)
  return x, y, h

def auto_pilot(intrinsic,
               disp,
               speed,
               look_dir,
               move_dir,
               position,
               camera_down,
               looklerp=0.05,
               movelerp=0.2,
               horizon=0.4,
               sky_fraction=0.1,
               near_fraction=0.2):
  """Auto-pilot algorithm that determines the next pose to sample.

  Args:
   intrinsic: Intrinsic matrix
   disp: disparity map
   speed: moving speed
   look_dir: look ahead direction
   move_dir: camera moving direction
   position: camera position
   camera_down: camera down vector (opposite of up vector)
   looklerp: camera viewing direction moving average interpolation ratio
   movelerp: camera translation moving average interpolation ratio
   horizon: predefined ratio of hozion in the image
   sky_fraction: predefined ratio of sky content
   near_fraction: predefined ratio of near content

  Returns:
   next_rot: next rotation to sample
   next_t: next translation to sample
   look_dir: next look ahead direction
   move_dir: next moving translation vector
   position: next camera position
  """
  img_w, img_h = disp.shape[-1], disp.shape[-2]

  x, y, h = skyline_balance(
      disp,
      horizon=horizon,
      sky_threshold=sky_fraction,
      near_fraction=near_fraction)
  look_uv = torch.stack([x, y], dim=-1)
  move_uv = torch.stack([torch.zeros_like(h) + 0.5, h], dim=-1)
  uvs = torch.stack([look_uv, move_uv], dim=1)

  fx, fy = intrinsic[0, 0], intrinsic[1, 1]
  px, py = intrinsic[0, 2], intrinsic[1, 2]
  c_x = (uvs[Ellipsis, 0] * img_w - px) / fx
  c_y = (uvs[Ellipsis, 1] * img_h - py) / fy
  new_coords = torch.stack([c_x, c_y], dim=-1)
  coords_h = torch.cat(
      [new_coords, torch.ones_like(new_coords[Ellipsis, 0:1])], dim=-1)

  new_look_dir = F.normalize(coords_h[:, 0, :], dim=-1)
  new_move_dir = F.normalize(coords_h[:, 1, :], dim=-1)
  look_dir = look_dir * (1.0 - looklerp) + new_look_dir * looklerp
  move_dir = move_dir * (1.0 - movelerp) + new_move_dir * movelerp
  move_dir_ = move_dir * speed.unsqueeze(-1)
  position = position + move_dir_
  # world to camera
  next_rot, next_t = pose_from_look_direction(
      position, look_dir.float(), camera_down)  # look from pos in direction dir
  return {
      'next_rot': next_rot,
      'next_t': next_t,
      'look_dir': look_dir,
      'move_dir': move_dir,
      'position': position
  }
  
  
def view_generation(warp_data, lerp, cam_speed):
    lerp = lerp#random.uniform(0.01, lerp)
    sky_fraction = 0.1
    near_fraction = 0.25
    use_auto_pilot = True
        
    cam_speed = cam_speed
    horizon = 0.4

    cur_disp = warp_data['disp'].squeeze(1)
        
    next_look_dir = torch.as_tensor(
        np.array([0.0, 0.0, 1.0]), dtype=torch.float32)
    next_look_dir = next_look_dir.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    next_move_dir = torch.as_tensor(
        np.array([0.0, 0.0, 1.0]), dtype=torch.float32)

    next_move_dir = next_move_dir.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    camera_down = torch.as_tensor(
        np.array([0.0, 1.0, 0.0]), dtype=torch.float32)
    camera_down = camera_down.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    speed = cam_speed * 7.5 * torch.ones(warp_data['img'].shape[0])

    cur_se3 = torch.eye(4).unsqueeze(0).repeat(cur_disp.shape[0], 1, 1)
    accumulate_se3 = torch.eye(4).unsqueeze(0).repeat(cur_disp.shape[0], 1, 1)
        
    # simple forward moving
    camera_pos = np.array([0.0, 0.0, 0.0])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])
    _, show_t = pose_from_look_direction_np(camera_pos, camera_dir,camera_up)
    t_c2_c1 = torch.as_tensor(show_t, dtype=torch.float32).unsqueeze(0)
    position = t_c2_c1.clone()
    # use auto-pilot to navigate automatically
    if use_auto_pilot:
      pilot_data = auto_pilot(
              warp_data['k_ref'][0].cpu(),
              cur_disp.detach().cpu(),
              speed,
              next_look_dir,
              next_move_dir,
              position,
              camera_down,
              looklerp=lerp,
              movelerp=lerp,
              horizon=horizon,
              sky_fraction=sky_fraction,
              near_fraction=near_fraction)

      next_rot = pilot_data['next_rot']
      next_t = pilot_data['next_t']
      next_look_dir = pilot_data['look_dir']
      next_move_dir = pilot_data['move_dir']
      position = pilot_data['position']
      # world to camera
      next_se3 = torch.cat([next_rot, next_t.unsqueeze(-1)], dim=-1)
      next_se3 = torch.cat(
          [next_se3, torch.zeros_like(next_se3[:, 0:1, :])], dim=1)
      next_se3[:, -1, -1] = 1.
      accumulate_se3 = next_se3

      camera_down = next_se3[:, 1, :3]
      # from current to next, delta pose
      fly_se3 = torch.matmul(next_se3, torch.inverse(cur_se3)) #.to(device)
      # update cur camera to world
      cur_se3 = next_se3
    return next_se3
    
if __name__ == '__main__':

    # ========================= load Parameters...====================
    # 
    path = "LHQ1024/lhq_1024/0000065.png"
    num_steps = 1
    fov_in_degrees = 55.
    lerp = 1.0  #random.uniform(0.01, lerp)
    sky_fraction = 0.1
    near_fraction = 0.25
    cam_speed = 0.08
    horizon = 0.4
    folder = 111
    slerp_nums = 100
    
    #diffusion inpaint
    pad_len = 128
    prompt = 'mountain'
    seed = 0
    scale = 10
    ddim_steps = 45
    # ========================= load model...=========================
    args = config.config_parser()
    device = "cuda:{}".format(args.local_rank)
    inpaint_config = "configs/v2-inpainting-inference.yaml"
    inpaint_ckpt = "pretrained/512-inpainting-ema.ckpt"
    # inpainter = Inpainter(args)
    inpainter = Diffusion_Inpainter(inpaint_config, inpaint_ckpt, device)
    # inpainter = None
    model = SpaceTimeModel(args)
    renderer = ImgRenderer(args, model, None, inpainter, device)
    model.switch_to_eval()
    dpt_model_path = 'third_party/DPT/weights/dpt_hybrid-midas-501f0c75.pt'
    dpt_model = DPTDepthModel(
        path=dpt_model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    dpt_model.eval()
    dpt_model.to(device)
    normalization = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = torchvision.transforms.Compose(
        [
            normalization
        ]
    )
    
    # ========================= load data...=========================
    
    img = imageio.v3.imread(path) / 255.
    img = cv2.resize(img, (512,512))[:,:,:3]
    img = torch.tensor(img.transpose(2,0,1))[None,Ellipsis]
    torchvision.utils.save_image(img,"img.png")

    with torch.no_grad():
      # p2d = (pad_len+5,pad_len+5,pad_len+5,0)
      # img_padded = F.pad(img[0,:3,5:img.shape[-2],5:(img.shape[-1]-5)], p2d, 'constant', 0.0).float().cuda()
      # mask = F.pad(torch.ones((((1,1,img.shape[-2]-5,img.shape[-1]-10)))), p2d, 'constant', 0.0).float().cuda()
      # outpaint_img = inpainter.inpaint(img_padded, mask, prompt, seed, scale, ddim_steps, 1, img_padded.shape[-1], img_padded.shape[-2])
      img_input = transform(img[0].cuda().float()).unsqueeze(0)
      prediction = dpt_model.forward(img_input)
      depth = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
      depth_min = depth.min()
      depth_max = depth.max()
      mono_disp = (depth - depth_min) / (depth_max - depth_min)

      px, py = (img.shape[-1] - 1) / 2., (img.shape[-2] - 1) / 2.
      
      fx , fy = img.shape[-1], img.shape[-2] #/ (2. * np.tan(fov_in_degrees / 360. * np.pi))
      k_ref = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]],dtype=np.float32)
      
      next_look_dir = torch.as_tensor(
          np.array([0.0, 0.0, 1.0]), dtype=torch.float32)
      next_look_dir = next_look_dir.unsqueeze(0).repeat(1, 1)

      next_move_dir = torch.as_tensor(
          np.array([0.0, 0.0, 1.0]), dtype=torch.float32)

      next_move_dir = next_move_dir.unsqueeze(0).repeat(1, 1)

      camera_down = torch.as_tensor(
          np.array([0.0, 1.0, 0.0]), dtype=torch.float32)
      camera_down = camera_down.unsqueeze(0).repeat(1, 1)

      speed = cam_speed * 7.5 * torch.ones(1)

      cur_se3 = torch.eye(4).unsqueeze(0).repeat(1, 1, 1)
      accumulate_se3 = torch.eye(4).unsqueeze(0).repeat(1, 1, 1)
      # simple forward moving
      camera_pos = np.array([0.0, 0.0, 0.0])
      camera_dir = np.array([0.0, 0.0, 1.0])
      camera_up = np.array([0.0, 1.0, 0.0])
      _, show_t = pose_from_look_direction_np(camera_pos, camera_dir,camera_up)
      t_c2_c1 = torch.as_tensor(show_t, dtype=torch.float32).unsqueeze(0)
      position = t_c2_c1.clone()
      output_path = os.path.join("output",path[-9:-4],'%d' %folder)
      os.makedirs(output_path, exist_ok=True)
      warp_data = {
          'img': img.float(),
          'disp': torch.tensor(mono_disp[None,None,Ellipsis]).float(),
          'k_ref': torch.tensor(k_ref)[None,Ellipsis].float()
            }  
      for i in range(num_steps):
    # ==================== load camera_pose...================
        cur_disp = warp_data['disp'].squeeze(1)
      
        pilot_data = auto_pilot(
            warp_data['k_ref'][0].cpu(),
            cur_disp.detach().cpu(),
            speed,
            next_look_dir,
            next_move_dir,
            position,
            camera_down,
            looklerp=lerp,
            movelerp=lerp,
            horizon=horizon,
            sky_fraction=sky_fraction,
            near_fraction=near_fraction)
  
        next_rot = pilot_data['next_rot']
        next_t = pilot_data['next_t']
        next_look_dir = pilot_data['look_dir']
        next_move_dir = pilot_data['move_dir']
        position = pilot_data['position']
        # world to camera
        next_se3 = torch.cat([next_rot, next_t.unsqueeze(-1)], dim=-1)
        next_se3 = torch.cat(
            [next_se3, torch.zeros_like(next_se3[:, 0:1, :])], dim=1)
        next_se3[:, -1, -1] = 1.
        accumulate_se3 = next_se3

        camera_down = next_se3[:, 1, :3]
        # from current to next, delta pose
        tgt_pose = torch.matmul(next_se3, torch.inverse(cur_se3)) #.to(device)
        # update cur camera to world
        cur_se3 = next_se3

        tgt_pose = tgt_pose.to(device)
        #######################################################
        p2d = (pad_len+5,pad_len+5,2*pad_len+5,0)
        img_padded = F.pad(img[0,:3,5:img.shape[-2],5:(img.shape[-1]-5)], p2d, 'constant', 0.0).float().cuda()
        mask = F.pad(torch.ones((((1,1,img.shape[-2]-5,img.shape[-1]-10)))), p2d, 'constant', 0.0).float().cuda()
        outpaint_img = inpainter.inpaint(img_padded, mask, prompt, seed, scale, ddim_steps, 1, img_padded.shape[-1], img_padded.shape[-2])
        outpaint_img = imageio.v3.imread('outpaint_img2.png') / 255.
        outpaint_img = torch.tensor(outpaint_img[:,:,:3].transpose(2,0,1))[None,Ellipsis].cuda().float()
        img_input = transform(outpaint_img[0]).unsqueeze(0)
        prediction = dpt_model.forward(img_input)
        depth = (
              torch.nn.functional.interpolate(
                  prediction.unsqueeze(1),
                  size=outpaint_img.shape[-2:],
                  mode="bicubic",
                  align_corners=False,
              )
              .squeeze()
              .cpu()
              .numpy()
          )
        depth_min = depth.min()
        depth_max = depth.max()
        outpaint_depth = (depth - depth_min) / (depth_max - depth_min)
        
        warp_data['outpaint_img'] = outpaint_img
        warp_data['outpaint_disp'] = torch.tensor(outpaint_depth[None,None,Ellipsis])
        
        if 'k_full' not in warp_data:
  
          px, py = (outpaint_img.shape[-1] - 1) / 2., (outpaint_img.shape[-2] - 1) / 2.
          fx, fy = outpaint_img.shape[-1], outpaint_img.shape[-2] #/ (2. * np.tan(fov_in_degrees / 360. * np.pi))
          k_full = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]],dtype=np.float32)
          warp_data['k_full'] = torch.tensor(k_full)[None,Ellipsis].float()
          
        #####################################################################
        renderer.tiny_process(warp_data)
        
        output_path_i = os.path.join(output_path, '%04d' % i)
        res_dict = renderer.render_rgbda_layers_from_one_view_flow(tgt_pose, output_path_i, slerp_nums, prompt, seed, scale, ddim_steps, dpt_model)
        warp_data['img'] = res_dict['pred_img']
        warp_data['disp'] = res_dict['pred_depth'].float().cpu()
        
        torchvision.utils.save_image(warp_data['img'],output_path_i + ".png")
        torchvision.utils.save_image(warp_data['disp'],output_path_i + "_depth.png")