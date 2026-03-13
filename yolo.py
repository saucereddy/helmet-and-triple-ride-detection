import numpy as np
import cv2 as cv
import os
from yoloDetection import detectObject, displayImage
import sys

global class_labels
global cnn_model
global cnn_layer_names

def loadLibraries(): #function to load yolov3 model weight and class labels
        global class_labels
        global cnn_model
        global cnn_layer_names
        class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n') #reading labels from yolov3 model
        print(str(class_labels)+" == "+str(len(class_labels)))
        cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights') #reading model
        
        out_layers = cnn_model.getUnconnectedOutLayers() #getting unconnected layers
        layer_names = cnn_model.getLayerNames() #getting all layers
        if isinstance(out_layers, np.ndarray) and out_layers.ndim > 1:
            cnn_layer_names = [layer_names[i[0] - 1] for i in out_layers]
        else:
            cnn_layer_names = [layer_names[i - 1] for i in out_layers]

def detectFromImage(imagename): #function to detect object from images
        label_colors = (0,255,0)
        try:
                image = cv.imread(imagename) #image reading
                if image is None:
                    print("Invalid image path")
                    return
                image_height, image_width = image.shape[:2]
        except Exception as e:
                print(f"Error reading image: {e}")
                return
        
        image, _, p_count, b_count = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, 0)
        displayImage(image, 0)

def detectFromVideo(videoFile): #function to read objects from video
        label_colors = (0,255,0)
        indexno = 0
        try:
                video = cv.VideoCapture(videoFile)
                frame_height, frame_width = None, None
        except:
                print("Unable to load video")
                return

        while True:
                frame_grabbed, frames = video.read()
                if not frame_grabbed:
                        break
                if frame_width is None or frame_height is None:
                        frame_height, frame_width = frames.shape[:2]
                
                frames, _, p_count, b_count = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels, indexno)
                cv.imshow("Video Detection", frames)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                indexno += 1

        print("Releasing resources")
        video.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
        loadLibraries()
        if len(sys.argv) == 3:
                if sys.argv[1] == 'image':
                        detectFromImage(sys.argv[2])
                elif sys.argv[1] == 'video':
                        detectFromVideo(sys.argv[2])
                else:
                        print("invalid input")
        else:
                print("sample commands to run code with image or video")
                print("python yolo.py image input_image_path")
                print("python yolo.py video input_video_path")
