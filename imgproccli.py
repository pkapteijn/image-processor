import logging
import sys
import executors as exc
from argparse import ArgumentParser

# Set up logger
def setupLogger(): 
    logger = logging.getLogger("video")
    FORMAT = '%(asctime)s:  %(message)s'
    logging.basicConfig(stream=sys.stdout, format=FORMAT,  level=logging.DEBUG)
    return logger

def parse_opts(): 
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="inputFilename",
                        help="input image or video filename", metavar="path/to/inputfilename", default=None)
    parser.add_argument("-s", "--show", help="show output image/video",
                        action="store_true")
    parser.add_argument("-m", "--mode", dest="mode",
                        help="image processing mode", choices=["flip", "detect", "classify", "blur", "haarcascade", "gray", "segmentation"], default="detect")
    parser.add_argument("-w", "--weights_model", dest="weightsModel",
                        help="weights data model to us for detect, classify and blur processing mode. \
                        If not specified default models will be used. ", default=None)
    return parser.parse_args()
    

def main(opts):
    logger.info(f"Commandline arguments: {opts}")

    # factory executor based on inputfilename
    factory = exc.ExecutorFactory()
    executor = factory.get_executor(opts, logger)

    # run the executor
    executor.run()

if __name__ == "__main__":
    opts = parse_opts()
    logger = setupLogger()
    main(opts)