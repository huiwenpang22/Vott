#import dependencies
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util


from os import listdir
from os.path import isfile, join

sys.path.append("..")


class object_detection:



    def __init__(self, num_classes ):
        self.num_classes = num_classes
        self.detection_graph = None
        self.frozen_graph_path = None
        self.label_map_path = None
        self.categories = None
        self.category_index = None
        self.serialized_graph = None
        self.label_map = None

        try:
            assert type(self.num_classes) is int
        except:
            print("Number of Classes must be an integer")


    @staticmethod


    def load_model(self, path_frozen_graph, label_map_path):
        self.frozen_graph_path = path_frozen_graph
        self.label_map_path = label_map_path
        try:
            assert tf.__version__ == "1.12.0"
        except:
            print("please make sure your tensorflow version is 1.12.0 this does not work with other versions")
        if not (os.path.isfile(path_frozen_graph) and os.path.isfile(label_map_path)):
            print("the paths to the graph and label map are invalid")
            return None



        try:
            print("loading frozen graph")
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.frozen_graph,'rb') as fid:
                    self.serialized_graph = fid.read()
                    od_graph_def.ParseFromString(self.serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
        except Exception as e:
            print("error while trying to load the model graph \n", e)

        print("frozen graph loaded successfully")
        print("loading label map")

        try:
            self.label_map = label_map_util.load_labelmap(label_map_path)
            self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
            self.category_index = label_map_util.create_category_index(self.categories)
        except Exception as e:
            print("error while trying to load the label map \n", e)
        print("label map loaded successfully")



    def infer_on_video(self, input_video_path, output_video_path, hide_display = True, min_score = .5):

        if not os.path.exists(input_video_path):
            print("the input file does not exist")
            return None


        # Checks and deletes the output file
        # You cant have a existing file or it will through an error
        if os.path.isfile(output_video_path):
            print("a file already exists with same name at {} \n, deleting that file".format(output_video_path))
            os.remove(output_video_path)


        # Playing video from file
        cap = cv2.VideoCapture(input_video_path)

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
        out = cv2.VideoWriter(output_video_path, fourcc, 10, (frame_width, frame_height))

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                while (cap.isOpened()):
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=5,
                        min_score_thresh=min_score)

                    if ret == True:
                        # Saves for video
                        out.write(frame)

                        # Display the resulting frame
                        if not hide_display:
                            cv2.imshow('Boiler Detection', frame)

                        # Close window when "Q" button pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break

                # When everything done, release the video capture and video write objects
                cap.release()
                out.release()

                # Closes all the frames
                cv2.destroyAllWindows()

if __name__=="__main__":
    boiler = object_detection(num_classes=1)
    boiler.load_model(r'Y:\boiler\frozen_inference_graph.pb',\
                      'Y:\boiler/pascal_label_map.pbtxt')
    boiler.infer_on_video(r'Y:\boiler\166_0010.MOV', r'Y:\boiler\new_video.mp4', \
                          hide_display=False,\
                          min_score=.8)

