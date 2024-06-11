# Lenovo Server에서 NOX 모델 사용
## using docker images : nvcr.io/nvidia/pytorch:23.01-py3
Lenovo Server에서 sample 영상 테스트나 NOX 모델 성능 테스트를 해보기 위해 input_video를 넣고 out_video로 저장시켜주는 코드

## Build

### 1. Docker Build
```Dockerfile
 docker run \
        -it \
        -p 12222:8888 \
        --name geon-yolo \
        --shm-size=6G \
        --runtime nvidia \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        nvcr.io/nvidia/pytorch:23.01-py3
```

### 2. ffmeng Install
```Dockerfile
git clone https://gitlab.inbic.duckdns.org/Dev-1-team/python-vms.git
```
### ffmeng 설치까지 끝났다면 
```Dockerfile
git clone -b argpurse https://gitlab.inbic.duckdns.org/Dev-1-team/ai-video-converter.git
```
### 도커에서 local로 나와 xhost build
```Dockerfile
xhost +
echo $DISPLAY
```
### 결과에 따라  :0,:1,:2 인지 확인 도커 안들으로 들어와서 
```Dockerfile
export DISPLAY = :2 ##echo 값에 따라 다름
```
### 코드 실행
```
python main.py --result_dir outputs/ --output_name result --nox_path models/NOX.pth --input inputs/0001.jpg 
#  --input 이미지, 동영상 둘다 들어가서 처리됨
# --concatenate true 하면 원본 이미지 옆 nox 이미지로 나옴
```

