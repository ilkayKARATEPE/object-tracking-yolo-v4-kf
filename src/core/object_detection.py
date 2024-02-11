import cv2
import numpy as np
import os


class ObjectDetectionYoloV4:
    def __init__(
        self,
        dnn_model_path,
        nms_threshold,
        conf_threshold,
        image_size,
    ):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self._dnn_model_path = dnn_model_path
        self._nms_threshold = nms_threshold
        self._conf_threshold = conf_threshold
        self._image_size = image_size

        # Load Network
        net = cv2.dnn.readNet(
            os.path.join(self._dnn_model_path, "yolov4.weights"),
            os.path.join(self._dnn_model_path, "yolov4.cfg"),
        )

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(
            size=(self._image_size, self._image_size), scale=1 / 255
        )

    def load_class_names(
        self,
    ):
        with open(
            os.path.join(self._dnn_model_path, "classes.txt"), "r"
        ) as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(
            frame, nmsThreshold=self._nms_threshold, confThreshold=self._conf_threshold
        )
