import torch
import torch.nn as nn
import torch.nn.functional as F

#mixed precision
from torch.cuda.amp import autocast

import os

import time

import cv2

import numpy as np
from utils import *

import sys
#import matplotlib.pyplot as plt

sys.path.append("../")

from utils import video_loader

class NOX_converter(video_loader.video_loader):
    def __init__(self, video_input, result_dir, video_output, model_path, showing=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        super().__init__(video_input, result_dir, video_output, showing)


    def model_loader(self,):
        _model = TransposeNoxModel(self.model_path,self.device)
        return _model

    def model_eval(self,image_batch, verbose=False):
        return self.nox_eval(image_batch,verbose)

    def nox_eval(self, image_batch, verbose=False):
        device = self.device
        # Convert frame
        data_lowlight_tensor = frame2tensor(image_batch)
        data_lowlight = data_lowlight_tensor.to(device)

		# Run model inference
        start = time.time()
        # enhanced_tensor (batch, h, w, c)
        with torch.no_grad():
            with autocast():
                enhanced_tensor = self.model.eval()(data_lowlight)
                if verbose:
                    end_time = (time.time() - start)
                    print(end_time)

                # Tensor to numpy and transpose (h, w, c)
                enhanced_frame = enhanced_tensor.data.to('cpu').numpy()

        return enhanced_frame
        


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Traditional Convolution
class TC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TC, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, input):
        out = self.conv(input)
        return out



# Depthwise Separable Convolution
class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class enhance_net_nopool(nn.Module):

    def __init__(self, conv_type='dsc'):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        number_f = 32

        # Define Conv type
        if conv_type == 'dsc':
            self.conv = DSC
        elif conv_type == 'dc':
            self.conv = DC
        elif conv_type == 'tc':
            self.conv = TC
        else:
            print("conv type is not available")

        #   zerodce DWC + p-shared
        self.e_conv1 = self.conv(3, number_f)

        self.e_conv2 = self.conv(number_f, number_f)
        self.e_conv3 = self.conv(number_f, number_f)
        self.e_conv4 = self.conv(number_f, number_f)

        self.e_conv5 = self.conv(number_f * 2, number_f)
        self.e_conv6 = self.conv(number_f * 2, number_f)
        self.e_conv7 = self.conv(number_f * 2, 3)

    def enhance(self, x, x_r):

        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return enhance_image

    def forward(self, x):
        x_down = x
        
        # extraction
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        x_r = x_r

        # enhancement
        enhance_image = self.enhance(x, x_r)

        #return enhance_image, x_r
        return enhance_image
    


def ReadFrame(cap):
    _, orig_img = cap.read()
    if orig_img is None:
        return False
    img_RGB = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    return img_RGB

def MultipleFrameBatch(cap, NumFrame=4):
    frameList = []
    for _ in range(NumFrame):
        _frameRGB = ReadFrame(cap)
        if _frameRGB is not False:
            frameList.append(_frameRGB)
        else:
            return np.array(frameList)
    return np.array(frameList)
    

def ConvertFrames(frame):
    frame_int = frame*255.0
    frame_int = cv2.convertScaleAbs(frame_int)
    frame_BGR = cv2.cvtColor(frame_int, cv2.COLOR_RGB2BGR)
    return frame_BGR

def BatchFramesConvert(video_out, batchFrame):
    frame_list = []
    for _frame in batchFrame:
        _cvrtFrame = ConvertFrames(_frame)
        video_out.write(_cvrtFrame)
    pass
        
    
    


def frame2batch(frame, margin=64):
    org_frame = frame
    hight, width, _ = frame.shape
    assert not (hight % 2 or width % 2), f'cannot divide hight({hight}) or width({width})'
        
    split_h = hight // 2
    split_w = width // 2
    
    # 1  2
    # 3  4
    split_1 = org_frame[0:split_h, 0:split_w, :]
    split_2 = org_frame[0:split_h, split_w:, :]
    split_3 = org_frame[split_h:, 0:split_w, :]
    split_4 = org_frame[split_h:, split_w:, :]
    
    batch_frame = np.stack((split_1, split_2, split_3, split_4),axis=0)
    
    return batch_frame

def batch2frame(batch_numpy, margin=64):
    q1, q2, q3, q4 = batch_numpy
    top = np.hstack((q1, q2))
    bottom = np.hstack((q3, q4))
    return np.vstack((top, bottom))
    


def frame2tensor(frame):
    data_lowlight = (np.asarray(frame) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    return data_lowlight
    

def ImageTranspose2Pytorch(data_lowlight, device):
    if data_lowlight.dim() < 4:
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = data_lowlight.to(device)
        data_lowlight = data_lowlight.unsqueeze(0)
    else:
        data_lowlight = data_lowlight.permute(0,3, 1, 2)
        data_lowlight = data_lowlight.to(device)
    return data_lowlight

def TensorTranspose2numpy(enhanced_tensor, device='cpu'):
    enhanced_tensor = enhanced_tensor.permute(0,2, 3, 1).squeeze()
    enhanced_image = enhanced_tensor.data.to(device).numpy()
    return enhanced_image

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
    
    
    
class TransposeChannel(nn.Module):
    def __init__(self, lastChannel=False):
        super(TransposeChannel, self).__init__()
        if lastChannel is True:
            self.transposeShape = (0,2,3,1)
        else:
            self.transposeShape = (0,3,1,2)
    def forward(self, input):
        return input.permute(self.transposeShape)
    
    
class TransposeNoxModel(nn.Module):
    def __init__(self, sgz_path = './Epoch99.pth', device='cuda'):
        super(TransposeNoxModel, self).__init__()
        self.org_net = enhance_net_nopool(conv_type='dsc').to(device)
        # Model load
        self.org_net.load_state_dict(torch.load(sgz_path, map_location=device))
        
        self.ChannelFirst = TransposeChannel(lastChannel=False)
        self.ChannelLast  = TransposeChannel(lastChannel=True)
        
        
    def forward(self, x):
        x = self.ChannelFirst(x)
        x = self.org_net(x)
        x = self.ChannelLast(x)
        return x



class Tester(): 
    def __init__(self, model_weight):
        #self.net = enhance_net_nopool(conv_type='dsc').to(device)
        # Model load
        #self.net.load_state_dict(torch.load(model_weight, map_location=device))
        self.net = TransposeNoxModel(model_weight,device)
        
        
    def inference_mixed_precision(self, frame, verbose=None):
        # Convert frame
        data_lowlight_tensor = frame2tensor(frame)
        data_lowlight = data_lowlight_tensor.to(device)

		# Run model inference
        start = time.time()
        # enhanced_tensor (batch, h, w, c)
        with torch.no_grad():
            with autocast():
                enhanced_tensor = self.net.eval()(data_lowlight)
                end_time = (time.time() - start)
                if verbose: print(end_time)

                # Tensor to numpy and transpose (h, w, c)
                enhanced_frame = enhanced_tensor.data.to('cpu').numpy()

        return enhanced_frame


    def video(self,video_input, result_dir, video_output):
        ## Read video
        cap = cv2.VideoCapture(video_input)
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        org_w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        org_h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ## Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(f"{result_dir}/{video_output}", fourcc, org_fps,(org_w,org_h))

        # read frames
        try:
            while True:
                #_, orig_img = cap.read()
                #if orig_img is None:
                #    break
                #img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                # make frame batch
                img_batch = MultipleFrameBatch(cap, NumFrame=1)
                if len(img_batch) == 0:
                    break

                # inference
                enhanced_numpy = self.inference_mixed_precision(img_batch,verbose=True)
                print(enhanced_numpy.shape)
                
                # combine batch to image
                
                #enhanced_frame = enhanced_numpy*255.0
                #result_frame = cv2.convertScaleAbs(enhanced_frame)
                #result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                #video_out.write(result_frame)
                BatchFramesConvert(
                    video_out=video_out,
                    batchFrame=enhanced_numpy,
                )
                

        finally:
            cap.release()
            video_out.release()
            cv2.destroyAllWindows()
            
            
            
if __name__ == '__main__':
    WEIGHT_DIR = '../models/NOX.pth'
    INPUT_PATH = '../inputs/sample1_1080.mp4'
    OUTPUT_DIR = '../outputs'
    OUTPUT_NAME = 'sample1_1080_trans_out.mp4'
    #t = Tester(model_weight=WEIGHT_DIR)
    #t.video(
    #    video_input = INPUT_PATH,
    #    result_dir = OUTPUT_DIR,
    #    video_output = OUTPUT_NAME,
    #)

    v_loader = NOX_converter(
            video_input     = INPUT_PATH,
            result_dir      = OUTPUT_DIR,
            video_output    = OUTPUT_NAME,
            model_path      = WEIGHT_DIR,
            showing = True,
        )
    v_loader.video_run(save=False)

