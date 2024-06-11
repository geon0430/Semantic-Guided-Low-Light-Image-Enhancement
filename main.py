
from utils.video_loader import *
from utils import nox

import argparse

import numpy as np
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NOX_converter(nox.NOX_converter):
    def __init__(self,
                 input_dir,
                 result_dir,
                 nox_path,
                 apply_nox=True,  
                 showing=False    
                 ):
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.nox_path = nox_path
        self.apply_nox = apply_nox
        self.showing = showing 

    def process_files(self):
        if not os.path.isdir(self.input_dir):
            print("Don't search inputs folder")
            return
        if not os.path.isdir(self.result_dir):
            print("Don't search outputs folder")
            return
        
        file_list = os.listdir(self.input_dir)
        if not file_list: 
            print("No files found in the input directory. Please upload files.")
            return

        for file_name in file_list:
            self.input_path = os.path.join(self.input_dir, file_name)
            extension = os.path.splitext(self.input_path)[1].lower()
            
            if extension != '.jpg':
                print(f"Please only place .mp4 files in the input directory. {file_name} is not a .mp4 file.")
                continue  

            self.output_name = os.path.splitext(file_name)[0] + "_NOX"
            super().__init__(self.input_path, self.result_dir, self.output_name, self.nox_path, self.showing)
            self.video_run()
    
    def video_run(self, save=True, concatenate=False):
        video_out = None 
        extension = os.path.splitext(self.input_path)[1].lower()

        if extension in ['.jpg', '.jpeg', '.png']:
            img_BGR = cv2.imread(self.input_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            output_BGR = img_BGR 
            if self.apply_nox:
                output_BGR = self.apply_models(img_RGB, verbose=True)

            if save:
                final_output = np.hstack((img_BGR, output_BGR)) if concatenate else output_BGR
                cv2.imwrite(f"{self.result_dir}/{self.output_name}.jpg", final_output)
            return

        try:
            cap = cv2.VideoCapture(self.input_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.input_path}")

            org_fps = cap.get(cv2.CAP_PROP_FPS)
            org_w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            org_h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if save:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_shape = (org_w * 2, org_h) if concatenate else (org_w, org_h)
                video_out = cv2.VideoWriter(f"{self.result_dir}/{self.output_name}.mp4", fourcc, org_fps, out_shape)

            err_counter = 0
            err_threshold = 60
            while True:
                img_RGB, img_BGR = ReadFrame(cap)
                if img_RGB is False:
                    err_counter += 1
                    if err_counter > err_threshold:
                        break
                    else:
                        continue
                
                output_BGR = img_BGR 
                if self.apply_nox:
                    output_BGR = self.apply_models(img_RGB, verbose=True)

                frame_BGR = np.hstack((img_BGR, output_BGR)) if concatenate else output_BGR

                if self.showing:
                    cv2.imshow('Monitor', frame_BGR)
                    if cv2.waitKey(1) != -1:
                        break
                if save:
                    video_out.write(frame_BGR)
                err_counter = 0
        except ValueError as e:
            print(e)  
        finally:
            cap.release()
            if save and video_out is not None: 
                video_out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./../../volume/LOLdataset/our485/low")
    parser.add_argument("--result_dir", default="./outputs", help="Directory to save the results")
    parser.add_argument("--nox_path", default="./models/Epoch97_1e4_6490.pth", help="Path to NOX model")
    parser.add_argument("--concatenate", type=str2bool, default=False, help="Whether to concatenate the original and processed images/videos")
    args = parser.parse_args()

    v_loader = NOX_converter(
        input_dir=args.input_dir,
        result_dir=args.result_dir,
        nox_path=args.nox_path,
    )
    v_loader.process_files()