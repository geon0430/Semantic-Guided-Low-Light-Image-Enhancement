import cv2
from pathlib import Path
from skimage import metrics
import re
import matplotlib.pyplot as plt
import os
import pandas as pd


def compare_ssim(img1, img2, use_grayscale=True, win_size=3):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    img1 = img1[0].detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2[0].detach().cpu().numpy().transpose(1, 2, 0)

    if use_grayscale:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return metrics.structural_similarity(img1, img2, win_size=win_size, data_range=img1.max() - img1.min(), full=True)[0]
    else:
        return metrics.structural_similarity(img1, img2, multichannel=True, win_size=win_size, data_range=img1.max() - img1.min(), full=True)[0]


def save_ssim_to_csv(epoch, average_ssim, filename="ssim_average.csv"):
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=['Epoch', 'Average SSIM'])

    new_data = pd.DataFrame({'Epoch': [int(epoch)], 'Average SSIM': [f"{average_ssim:.4f}"]})
    data = pd.concat([data, new_data], ignore_index=True)

    data.to_csv(filename, index=False, float_format='%.4f')

