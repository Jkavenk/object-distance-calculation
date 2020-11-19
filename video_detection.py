import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv2
import time

# Import object detection module
from utils import label_map_util
from utils import visualization_utils as vis_util

#import text to speech library
import pyttsx3

#initialization Text To Speech
engine = pyttsx3.init()

#text-to-speech articulation speed
rate = engine.getProperty('rate')
engine.setProperty('rate', 125)
#set the text-to-speech voice to women[1] and man[0]
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

VIDEO_OUTPUT = 'test object detection.avi'

# Error handling
if os.path.isfile(VIDEO_OUTPUT):
    os.remove(VIDEO_OUTPUT)

# read video from file
cap = cv2.VideoCapture('demo alat sign.mp4')
# take live-feed from webcam or external camera
#cap = cv2.VideoCapture(0)

# convert resolution from float to integer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))

sys.path.append("..")

MODEL_NAME = 'trained-inf-final'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('annotations', 'label_map.pbtxt')
NUM_CLASSES = 4

# load frozen tensorflow model to memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#error handling for TTS
apx_distance1 = 0
#counter TTS
a = 1

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        while(cap.isOpened()):    
            ret, frame = cap.read()
            
            if ret == True:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)
                class_detected = 0
            # run object detection
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

            # object detection visualization
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.7,
                    line_thickness=4)
            
            # calibrate distance
                for i,b in enumerate(boxes[0]):
                    # tree class = 1
                    if classes[0][i] == 1:
                        if scores[0][i] >= 0.75:
                            class_detected = 1
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            large = ((boxes[0][i][3] - boxes[0][i][1])*(boxes[0][i][2] - boxes[0][i][0]))
                            a+=1
                            #distance 5.9 meters
                            if large >= 0.3174: # and large < 0.514:
                                kdistance = 5.9
                                klarge = 0.4706
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                            #distance 8.9 meters
                            if large >= 0.1774 and large < 0.3174:
                                kdistance = 8.9
                                klarge = 0.2206
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                                apx_distance1 = apx_distance
                            
                            #distance 11.9 meters
                            if large >= 0.127045 and large < 0.1774:
                                kdistance = 11.9
                                klarge = 0.1447
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                                apx_distance1 = apx_distance
                            
                            #distance 14.9 meters
                            if large >= 0.08784 and large < 0.127045:
                                kdistance = 14.9
                                klarge = 0.103229
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                            print("tree: ",apx_distance)
                                    
                # NO STOP SIGN class = 2
                    if classes[0][i] == 2:
                        if scores[0][i] >= 0.8:
                            class_detected = 2
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            large = ((boxes[0][i][3] - boxes[0][i][1])*(boxes[0][i][2] - boxes[0][i][0]))
                            # distance 15 meters
                            if large < 0.00462:
                                kdistance = 15
                                klarge = 0.00314
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 12 meters
                            elif large >= 0.00462 and large < 0.00692:
                                kdistance = 12
                                klarge = 0.004891
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 9 meters
                            elif large >= 0.00692 and large < 0.01532:
                                kdistance = 9
                                klarge = 0.00861
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 6 meters
                            elif large >= 0.01532 and large < 0.06734:
                                kdistance = 6
                                klarge = 0.0212
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 3 meters
                            elif large >= 0.06734:
                                kdistance = 3
                                klarge = 0.09456
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                            a+=1
                            print("no stop sign: ", apx_distance)
                                
                # SPEED WARNING class = 3
                    if classes[0][i] == 3:
                        if scores[0][i] >= 0.8:
                            class_detected = 3
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            large = ((boxes[0][i][3] - boxes[0][i][1])*(boxes[0][i][2] - boxes[0][i][0]))
                            # distance 15 meters
                            if large < 0.00462:
                                kdistance = 15
                                klarge = 0.00314
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 12 meters
                            elif large >= 0.00462 and large < 0.00692:
                                kdistance = 12
                                klarge = 0.004891
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 9 meters
                            elif large >= 0.00692 and large < 0.01532:
                                kdistance = 9
                                klarge = 0.00861
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 6 meters
                            elif large >= 0.01532 and large < 0.06734:
                                kdistance = 6
                                klarge = 0.0212
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                
                                # distance 3 meters
                            elif large >= 0.06734:
                                kdistance = 3
                                klarge = 0.09456
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                apx_distance1 = apx_distance
                            a+=1
                            print("speed warning sign: ", apx_distance)
                            
                 # Traffic light class = 4
                    if classes[0][i] == 4:
                        if scores[0][i] >= 0.7:
                            class_detected = 4
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            w = (boxes[0][i][1]+boxes[0][i][3])
                            h = (boxes[0][i][0]+boxes[0][i][2])
                            large = ((boxes[0][i][3] - boxes[0][i][1])*(boxes[0][i][2] - boxes[0][i][0]))
                                # distance 2.9 meters
                            if large >= 0.02601: 
                                kdistance = 2.9
                                klarge = 0.0462
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                # distance 6.1 meters
                            if large >= 0.00704 and large < 0.02601:
                                kdistance = 6.1
                                klarge = 0.01018
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                                apx_distance1 = apx_distance
                                # distance 9.1 meters
                            if large < 0.00704:
                                kdistance = 9.1
                                klarge = 0.00471
                                apx_distance = round((klarge * kdistance) / large,2)
                                cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                                apx_distance1 = apx_distance
                            a+=1
                            print("traffic light: ",apx_distance)
                
                # save frame into video
                out.write(frame)

                # shows frame
                cv2.imshow('Object Detection', cv2.resize(frame,(400,700)))
                
                #Text-To-Speech function
                '''if a%30 == 0:
                    if class_detected == 1:
                        engine.say('tree in {}meters'.format(apx_distance1))
                        
                    elif class_detected == 2 or class_detected == 3:
                        engine.say('traffic sign in {}meters'.format(apx_distance1))
                        
                    elif class_detected == 4:
                        engine.say('traffic light in {}meters'.format(apx_distance1))
                
                engine.runAndWait()'''
                
                # close window with q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
