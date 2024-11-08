import processors as prcs
import os
import time
import sys
import cv2 as cv
import numpy as np
from alive_progress import alive_bar

# supported input formats
VIDEO_EXT = ("mp4")
IMAGE_EXT = ("jpg", "jpeg", "png", "tiff")

class ExecutorFactory: 

    def get_executor(self, opts, logger): 
        if self._isVideo(opts): 
            return VideoExecutor(opts, logger)
        elif self._isImage(opts): 
            return ImageExecutor(opts, logger)
        elif self._isWebcam(opts): 
            return WebcamExecutor(opts, logger)
        else: 
            logger.error(f"Could not open input file {opts.inputFilename if opts.inputFilename != None else 'webcam'}")
            logger.info(f"Supported file extensions: {VIDEO_EXT}, {IMAGE_EXT}")
            sys.exit()

    def _isVideo(self, opts): 
        ext = opts.inputFilename.split("/")[-1].split(".")[-1]
        if ext in VIDEO_EXT: 
            return True
        else: 
            return False

    def _isImage(self, opts): 
        ext = opts.inputFilename.split("/")[-1].split(".")[-1]
        if ext in IMAGE_EXT: 
            return True
        else: 
            return False

    def _isWebcam(self, opts): 
        if opts.inputFilename == None: 
            return True
        else: 
            return False

class Executor: 
    def __init__(self, opts, logger) -> None:
        self.opts = opts
        self.logger = logger
        
    def getMilliTime(self):
        return round(time.time() * 1000)

    def getOutputFilename(self):
        OUTDIRVIDEO = os.path.join(".", "video", "processed")
        OUTDIRIMG = os.path.join(".", "image", "processed")
        if not os.path.exists(OUTDIRVIDEO): 
            os.makedirs(OUTDIRVIDEO)
        if not os.path.exists(OUTDIRIMG): 
            os.makedirs(OUTDIRIMG)    
        outbase = self.opts.inputFilename.split("/")[-1].split(".")[0]
        ext = self.opts.inputFilename.split("/")[-1].split(".")[-1]
        if ext in VIDEO_EXT: 
            name = os.path.join(OUTDIRVIDEO, outbase + "-processed-"+ self.opts.mode + ".mp4")
        else:
            name = os.path.join(OUTDIRIMG, outbase + "-processed-"+ self.opts.mode + ".png") #ext

        return os.path.abspath(name)

class ImageExecutor(Executor): 
    def __init__(self, opts, logger) -> None:
        super().__init__(opts, logger)

    def run(self):
        startTime = super().getMilliTime()

         # factory processor based on mode
        factory = prcs.ProcessorFactory()
        processor = factory.get_processor(self.opts)

        logger = self.logger
                
        img = cv.imread(self.opts.inputFilename)
        if img is None:
            sys.exit("Could not read the image.")
        height, width, clrdepth = self._getImageDimensions(img)
        logger.info(f"Image height: {height}")
        logger.info(f"Image width: {width}")
        logger.info(f"Image color depth: {clrdepth}")

        prcimg = processor.process(img)
        outfile = super().getOutputFilename()
        cv.imwrite(outfile, prcimg)
        proctim = super().getMilliTime()-startTime
        logger.info(f"Total processing time: {proctim} ms")
        logger.info(f"Output written to: {outfile}")

        if self.opts.show: 
            win = "ProcessedFrame"
            cv.namedWindow(win, cv.WINDOW_NORMAL) 
            cv.imshow(win, prcimg) 
            logger.info(f"Press any key to exit")
            k = cv.waitKey(0)
        

    def _getImageDimensions(self, img): 
        (height, width, clrdepth) = img.shape
        return height, width, clrdepth
    

class VideoExecutor(Executor): 
    def __init__(self, opts, logger) -> None:
        super().__init__(opts, logger)

    def run(self):
        # factory processor based on mode
        factory = prcs.ProcessorFactory()
        processor = factory.get_processor(self.opts)

        logger = self.logger

        cap = cv.VideoCapture(self.opts.inputFilename) 
        if cap.isOpened(): 
            logger.info(f"Inputfile is opened: {cap.isOpened()}")
            width, height, fps, fcnt, length = self._getVideoDimensions(cap)
            logger.info(f"Frame rate [fps]: {round(fps, 2)}")
            logger.info(f"Frame count [#]: {int(fcnt)}")
            logger.info(f"Frame width [pix]: {int(width)}")
            logger.info(f"Frame height [pix]: {int(height)}")
            logger.info("Input video length[s]: {0}".format(round(length, 2)))        
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            outfile = super().getOutputFilename()
            out = cv.VideoWriter(outfile, fourcc, fps, (int(width),  int(height)))
            procTimes = []
        else: 
            logger.error(f"Could not open input file {self.opts.inputFilename}")
            sys.exit()

        with alive_bar(fcnt) as bar:
            # Loop through each videoframe
            while cap.isOpened():
                startTime = super().getMilliTime()
                ret, img = cap.read()
                if not ret:
                    logger.info("Stream end. Exiting ...")
                    break

                prcimg = processor.process(img)

                # write the processed frame
                out.write(prcimg)

                if self.opts.show: 
                    cv.imshow('Processedframe', prcimg)

                procTimes.append(super().getMilliTime()-startTime)
                bar()
                if cv.waitKey(1) == ord('q'):
                    break

        avg, std, tot = self._getProcessingStats(procTimes)
        logger.info(f"Average frame procesing time: {round(avg, 2)} ms")
        logger.info(f"Standard deviation frame procesing time: {round(std, 2)} ms")
        logger.info(f"Total processing time: {round(tot, 2)} s")
        logger.info(f"Real-time processing: {tot<length}")
        logger.info(f"Output written to: {outfile}")

        # Release everything if job is finished
        cap.release()
        out.release()

        
    def _getVideoDimensions(self, video): 
        if video.isOpened(): 

            width  = video.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = video.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
            fps = video.get(cv.CAP_PROP_FPS)
            fcnt = video.get(cv.CAP_PROP_FRAME_COUNT)
            length = fcnt/fps
        else: 
            width, height, fps, fcnt, length = 0
        return int(width), int(height), fps, int(fcnt), length

    def _getProcessingStats(self, timesArray): 
        procTimesNd = np.array(timesArray)
        avgProcTime = np.average(procTimesNd)
        totProcTime = np.sum(procTimesNd)/1000
        stdProcTime = np.std(procTimesNd)

        return avgProcTime, stdProcTime, totProcTime
    

class WebcamExecutor(Executor): 
    def __init__(self, opts, logger) -> None:
        super().__init__(opts, logger)

    def run(): 
        pass