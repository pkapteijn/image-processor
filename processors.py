import random
from ultralytics import YOLO
import cv2 as cv
import numpy as np

class ProcessorFactory: 
    def get_processor(self, opts): 
        if opts.mode == "detect": 
            return DetectProcessor(opts.weightsModel)
        elif opts.mode == "segmentation": 
            return SegmentationProcessor(opts.weightsModel)
        elif opts.mode == "classify": 
            return ClassifyProcessor(opts.weightsModel)
        elif opts.mode == "blur": 
            return BlurProcessor(opts.weightsModel)
        elif opts.mode == "haarcascade": 
            return HaarCascadeProcessor(opts.weightsModel)
        elif opts.mode == "gray": 
            return GrayProcessor()
        else: 
            return FlipProcessor()

class DetectProcessor:
    def __init__(self, model):
        if model != None: 
            self.model = YOLO(model)
        else: 
            self.model = YOLO("yolo11n.pt")
        self.classes = self.model.names
        self.colors = [tuple((random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255))) for _ in range(len(self.classes.values()))]

    
    def process(self, frame):
        # Run inference on an image
        results = self.model(frame, verbose=True)
        # Process results generator
        for result in results:
            boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
            cls = result.boxes.cls.tolist()
            confs = result.boxes.conf.tolist()
            
            for cnf, cl, box in zip(confs, cls, boxes):
                frame = self._drawDetectBox(frame, cnf, cl, box)   
        return frame
    
    def _drawDetectBox(self, frame, confidence, cl, box): 
        tl    = (int(box[0]), int(box[1]))
        tllbl = (int(box[0]), int(box[1])-6)
        br    = (int(box[2]), int(box[3]))
        bl    = (int(box[0])+3, int(box[3]-6))
        prc   = f"{confidence*100:.2f}%"
        color = tuple(self.colors[int(cl)])
        label = f"{self.classes[int(cl)]}"
        frame = cv.rectangle(frame, tl, br, color, 1)
        frame = cv.putText(
            frame, prc, bl, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        frame = cv.putText(
            frame, label, tllbl, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame

class SegmentationProcessor:
    def __init__(self, model):
        if model != None: 
            self.model = YOLO(model)
        else: 
            self.model = YOLO("yolo11n-seg.pt")
        self.classes = self.model.names
        self.colors = [tuple((random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255))) for _ in range(len(self.classes.values()))]

    
    def process(self, frame):
        # Run inference on an image
        results = self.model(frame, verbose=True)
        # Process results generator
        #print(f"results: {results[0].masks}") 
        for result in results: 
            frame = self._drawSegmentationShape(frame, result.masks)
        return frame
    
    def _drawSegmentationShape(self, frame, masks):
        for mask in masks:  
            pts = np.array(mask.xy, np.int32)
            pts = pts.reshape((-1,1,2))
            print(f"pts. shape: {pts.shape}")
            print(f"pts. dtype: {pts.dtype}")
            frame = cv.polylines(frame,pts,True,(0,255,255), 2)
        return frame

class ClassifyProcessor:
    def __init__(self, model):
        if model != None: 
            self.model = YOLO(model)
        else: 
            self.model = YOLO("yolo11n-cls.pt")
        self.classes = self.model.names
        self.colors = [tuple((random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255))) for _ in range(len(self.classes.values()))]

    def process(self, frame):
        # Run inference on an image
        results = self.model(frame)
        # Process results generator
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            org = (5, 30) # start coordinate for first label (topleft)
            for cl, cnf in zip(probs.top5, probs.top5conf): 
                frame, org = self._drawClassifyLabels(frame, org, cl, cnf)     
        return frame
    
    def _drawClassifyLabels(self, frame, org, cl, confidence):
        # filter out low confidence classifications <30% 
        if confidence > 0.3:
            label = f"{self.classes[int(cl)]}: {confidence*100:.2f}%"
            color = self.colors[int(cl)]
            textsize = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
            frame = cv.rectangle(frame, org, (org[0] + 450, org[1] - textsize[0][1] - 10), color, -1)
            frame = cv.putText(
                frame, label, (org[0]+5, org[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            # stack the next label below
            nextorg = (org[0], org[1] + textsize[0][1] + 15)
        else: 
            nextorg = org

        return frame, nextorg

class BlurProcessor:
    def __init__(self, model):
        if model != None: 
            self.model = YOLO(model)
        else: 
            self.model = YOLO("licplate-best.pt")
        self.classes = self.model.names
        self.colors = [tuple((random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255))) for _ in range(len(self.classes.values()))]

    
    def process(self, frame):
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Run inference on an image
        results = self.model(frame, verbose=True)
        # Process results generator
        for result in results:
            boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
            cls = result.boxes.cls.tolist()
            confs = result.boxes.conf.tolist()
            
            for cnf, cl, box in zip(confs, cls, boxes):
                frame = self._blurROI(frame, cnf, cl, box)   
        return frame
    
    def _blurROI(self, frame, confidence, cl, box): 
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])

        frame[y:y+h, x:x+w] = cv.blur(frame[y:y+h, x:x+w] ,(20, 20))
        return frame

class HaarCascadeProcessor:
    def __init__(self, model):
        if model != None: 
            self.model = model
        else: 
            self.model = "haarcascade_frontalface_default.xml"
        self.classifier = cv.CascadeClassifier(
            cv.data.haarcascades + self.model)
        self.colors = [tuple((random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255))) for _ in range(1)]

    
    def process(self, frame):
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            frame = self._drawBox(frame, x, y, w, h)  
        return frame
    
    def _drawBox(self, frame, x, y, w, h): 
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), self.colors[0], 4)  
        return frame

class FlipProcessor:
    def process(self, frame):  
        img = cv.flip(frame, 1) # 1 - vertical flip
        return img
    
class GrayProcessor:
    def process(self, frame):  
        img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        return img
    

"""
img = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=WHITE)
mg2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
instance segmentation yolo
"""