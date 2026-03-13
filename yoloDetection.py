import numpy as np
import cv2 as cv

def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels, indexno,  
            Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True):
    
    option = 0
    counter = 0
    if detect:
        blob_object = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        CNNnet.setInput(blob_object)
        cnn_outs_layer = CNNnet.forward(total_layer_names)
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(cnn_outs_layer, image_height, image_width, 0.5)
        ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, 0.5, 0.3)
        
        if Boundingboxes is None or confidence_value is None or ids is None or class_ids is None:
           print("[ERROR] unable to draw boxes.")
           return image, 0, 0
        
        image, option, p_count, b_count = labelsBoundingBoxes(image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels, indexno)
    else:
        p_count, b_count = 0, 0

    return image, option, p_count, b_count


def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels, indexno):
    option = 0
    p_count = 0
    b_count = 0
    if len(ids) > 0:
        # Compatibility for different NMSBoxes return types
        if isinstance(ids, np.ndarray):
            ids_list = ids.flatten()
        else:
            ids_list = ids

        for i in ids_list:
            xx, yy = Boundingbox[i][0], Boundingbox[i][1]
            width, height = Boundingbox[i][2], Boundingbox[i][3]
            
            class_color = (0, 255, 0)

            cv.rectangle(image, (xx, yy), (xx+width, yy+height), class_color, 2)
            
            # Usually class 0 is person and class 1 is motorbike in YOLOv3 COCO or similar
            if classID[i] <= 1:
                if classID[i] == 0: p_count += 1
                if classID[i] == 1: b_count += 1
                
                try:
                    label_name = predicted_labels[classID[i]]
                except IndexError:
                    label_name = "Object"
                
                text_label = "{}: {:.2f}".format(label_name, conf_thr[i])
                (label_w, label_h), baseline = cv.getTextSize(text_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.rectangle(image, (xx, yy - label_h - baseline), (xx + label_w, yy), class_color, -1)
                cv.putText(image, text_label, (xx, yy - baseline), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                option = 1

    return image, option, p_count, b_count


def listBoundingBoxes(outs, image_height, image_width, threshold_conf):
    box_array = []
    confidence_array = []
    class_ids_array = []

    for img in outs:
        for obj_detection in img:
            detection_scores = obj_detection[5:]
            class_id = np.argmax(detection_scores)
            confidence_value = obj_detection[4] * detection_scores[class_id]
            if confidence_value > threshold_conf and class_id <= 1:
                Boundbox = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_X, center_Y, box_width, box_height = Boundbox.astype('int')

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array

def displayImage(image, index):
    cv.imshow("Final Image", image)
    cv.waitKey(1)
