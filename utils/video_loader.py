import cv2
import numpy as np

from pathlib import Path

def ReadFrame(cap):
    _, orig_img = cap.read()
    if orig_img is None:
        return False, False
    img_RGB = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    return img_RGB, orig_img

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
    for _frame in batchFrame:
        _cvrtFrame = ConvertFrames(_frame)
        video_out.write(_cvrtFrame)
    pass



class video_loader:
    def __init__(self, video_input, result_dir, video_output, showing=False):
        self.video_input = video_input
        self.result_dir = result_dir
        self.video_output = video_output
        self.showing = showing

        self.check_file(video_input)
        
        self.model = self.model_loader()

    def check_file(self,file_name):
        path = Path(f"{file_name}")
        assert path.is_file(); f"File is not exist, {path.absolute()}"


    def model_loader(self,):
        pass

    def model_eval(self,image_batch):
        return image_batch


    def apply_models(self,image_frame, verbose=False):
        assert  image_frame.ndim == 3, f"Check input frame dims is 3 : {image_fram.ndim}"
        image_batch = np.expand_dims(image_frame, axis=0)

        # Apply deep learning models
        output_batch = self.model_eval(image_batch)


        output_image_frame = output_batch.squeeze()
        assert  output_image_frame.ndim == 3, f"Check output frame dims is 3 : {output_image_frame.ndim}"

        frame_BGR = ConvertFrames(output_image_frame)
        
        return frame_BGR


    def video_run(self, save = True):
        ## Read video
        cap = cv2.VideoCapture(self.video_input)
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        org_w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        org_h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ## Save video
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(f"{self.result_dir}/{self.video_output}", fourcc, org_fps,(org_w,org_h))

        # read frames
        err_counter = 0
        err_threshold = 60
        try:
            while True:
                img_RGB, img_BGR = ReadFrame(cap)
                if img_RGB is False:
                    print("Not Get Frame")
                    err_counter += 1
                    if err_counter > err_threshold:
                        break
                    else:
                        continue


                # inference
                frame_BGR = self.apply_models(img_RGB,verbose=True)
                print(frame_BGR.shape)

                if self.showing:
                    cv2.imshow('Monitor', frame_BGR)
                    if cv2.waitKey(1) != -1:
                        break    
                if save:
                    video_out.write(frame_BGR)
                err_counter = 0
                

        finally:
            cap.release()
            if save:
                video_out.release()
            cv2.destroyAllWindows()

