# image-processor
Runs image processing algorithms on image and video files. 
Supported algorithms: 
- Object detection (YOLO based)
- Image classification (YOLO based)
- Instance segmentation (YOLO based)
- Object blurring (YOLO based)
- Segmentation (haarcascade based)
- Vertical flip/mirror
- Color to grayscale

It accepts image and video files.  Currently supported(tested) file types: 
-.mp4
-.jpg,.jpeg,.png,.bmp

Output is written to the folders "./video/processed" or "./image/processed",  depending on the input(video or image)

Webcam processing is still WIP. 

## Installation

### Pre-requisites

- python3
- pip3

### Python packages

```
$ pip3 install opencv-python
$ pip3 install ultralytics
$ pip3 install numpy
```

## Running

```
$ python3 imgproccli.py -h
usage: imgproccli.py [-h] [-i path/to/inputfilename] [-s] [-m {flip,detect,classify,blur,haarcascade,gray,segmentation}] [-w WEIGHTSMODEL]

options:
  -h, --help            show this help message and exit
  -i path/to/inputfilename, --input path/to/inputfilename
                        input image or video filename
  -s, --show            show output image/video
  -m {flip,detect,classify,blur,haarcascade,gray,segmentation}, --mode {flip,detect,classify,blur,haarcascade,gray,segmentation}
                        image processing mode
  -w WEIGHTSMODEL, --weights_model WEIGHTSMODEL
                        weights data model to us for detect, classify and blur processing mode. If not specified default models will be used.
                        
```


# TODO

- API wrapper & containerise, next to CLI
- add other algos
- allow more than one algo to be applied in sequence in one run
- webcam processing
- put each processor in a single module
