import warnings
from threading import Event

warnings.filterwarnings("ignore")
import time
import cv2
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm

import EmotionRecognition.EmotionDeepFace as EmotionDeepFace
import EmotionRecognition.functions as functions

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)

class_indices = EmotionDeepFace.getClassIndices()


def build_model(model_name, modelPath):
    """
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes
	Returns:
		built deepface model
	"""

    global model_obj  # singleton design pattern

    models = {
        'EmotionDeepFace': EmotionDeepFace.loadModel
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model(
                path=modelPath
            )
            model_obj[model_name] = model
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]


def analyze(img_path, actions=('emotion', 'age', 'gender', 'race'), model=None, enforce_detection=True,
            detector_backend='opencv', prog_bar=False, modelPath=''):
    """
	This function analyzes facial attributes including age, gender, emotion and race

	Parameters:
		img_path: exact image path, numpy array or base64 encoded image could be passed. If you are going to analyze lots of images, then set this to list. e.g. img_path = ['img1.jpg', 'img2.jpg']

		actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop some of those attributes.

		models: (Optional[dict]) facial attribute analysis models are built in every call of analyze function. You can pass pre-built models to speed the function up.

			models = {}
			models['age'] = DeepFace.build_model('Age')
			models['gender'] = DeepFace.build_model('Gender')
			model = DeepFace.build_model('Emotion')
			models['race'] = DeepFace.build_model('Race')

		enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib.

		prog_bar (boolean): enable/disable a progress bar
	Returns:
		The function returns a dictionary. If img_path is a list, then it will return list of dictionary.

		{
			"region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
			"age": 28.66,
			"gender": "woman",
			"dominant_emotion": "neutral",
			"emotion": {
				'sad': 37.65260875225067,
				'angry': 0.15512987738475204,
				'surprise': 0.0022171278033056296,
				'fear': 1.2489334680140018,
				'happy': 4.609785228967667,
				'disgust': 9.698561953541684e-07,
				'neutral': 56.33133053779602
			}
			"dominant_race": "white",
			"race": {
				'indian': 0.5480832420289516,
				'asian': 0.7830780930817127,
				'latino hispanic': 2.0677512511610985,
				'black': 0.06337375962175429,
				'middle eastern': 3.088453598320484,
				'white': 93.44925880432129
			}
		}

	"""

    actions = list(actions)

    img_paths, bulkProcess = functions.initialize_input(img_path)

    # ---------------------------------

    built_models = ["emotion"] if model != None else []
    # ---------------------------------

    # pre-trained models passed but it doesn't exist in actions
    if len(built_models) > 0:
        if 'emotion' in built_models and 'emotionDeepFace' not in actions:
            actions.append('emotionDeepFace')

    # ---------------------------------
    if 'emotionDeepFace' in actions and 'emotion' not in built_models:
        model = build_model('EmotionDeepFace', modelPath)
    # ---------------------------------

    resp_objects = []

    disable_option = (False if len(img_paths) > 1 else True) or not prog_bar

    global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable=disable_option)

    for j in global_pbar:
        img_path = img_paths[j]

        resp_obj = {}

        disable_option = (False if len(actions) > 1 else True) or not prog_bar

        pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable=disable_option)

        img_224 = None  # Set to prevent re-detection

        region = []  # x, y, w, h of the detected face region
        region_labels = ['x', 'y', 'w', 'h']

        is_region_set = False

        # facial attribute analysis
        for index in pbar:
            action = actions[index]
            pbar.set_description("Action: %s" % (action))

            emotion_labels = class_indices
            img, region = functions.preprocess_face(img=img_path, target_size=(48, 48), grayscale=True,
                                                    enforce_detection=enforce_detection,
                                                    detector_backend=detector_backend, return_region=True)

            if img == []:
                resp_obj["emotion"] = {}
                for i in range(0, len(emotion_labels)):
                    emotion_label = emotion_labels[i]
                    emotion_prediction = 0

                resp_obj["dominant_emotion"] = "Not Found"
                break

            emotion_predictions = model.predict(img)[0, :]

            sum_of_predictions = emotion_predictions.sum()

            resp_obj["emotion"] = {}

            for i in range(0, len(emotion_labels)):
                emotion_label = emotion_labels[i]
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                resp_obj["emotion"][emotion_label] = emotion_prediction

            resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

            # -----------------------------

            if is_region_set != True:
                resp_obj["region"] = {}
                is_region_set = True
                for i, parameter in enumerate(region_labels):
                    resp_obj["region"][parameter] = int(
                        region[i])  # int cast is for the exception - object of type 'float32' is not JSON serializable

            return resp_obj
        # ---------------------------------

        if bulkProcess == True:
            resp_objects.append(resp_obj)
        else:
            return resp_obj

    if bulkProcess == True:

        resp_obj = {}

        for i in range(0, len(resp_objects)):
            resp_item = resp_objects[i]
            resp_obj["instance_%d" % (i + 1)] = resp_item

        return resp_obj


# ---------- gui.py variables initialization and thread event ----------
stop_main_emotions = Event()
emotionsCounter = {"angry": 0,
                   "disgust": 0,
                   "fear": 0,
                   "happy": 0,
                   "sad": 0,
                   "surprise": 0,
                   "neutral": 0}


def stopEmotions():
    stop_main_emotions.set()  # stops emotion recognition thread

    # ---------- Average emotion of the last music ----------
    return max(emotionsCounter, key=emotionsCounter.get)


def main(window):
    # ---------------------------
    # main

    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing import image

    musicTime = 6
    video = cv2.VideoCapture(0)

    # ---------- gui.py variables initialization ----------
    stop_main_emotions.clear()

    global emotionsCounter
    emotionsCounter = {"angry": 0,
                       "disgust": 0,
                       "fear": 0,
                       "happy": 0,
                       "sad": 0,
                       "surprise": 0,
                       "neutral": 0}

    while True:
        _, frame = video.read()

        result = analyze(
            frame,
            detector_backend="mtcnn",  # opencv, ssd, dlib, mtcnn, retinaface, mediapipe
            actions=['emotionDeepFace'],
            modelPath='weights/facial_expression_model_weights.h5'
        )

        print(result)

        # ---------- Round emotions values ----------
        percentages = ''
        for emotion in result['emotion']:
            percentages += str(round(result['emotion'][emotion], 3))
            if emotion != 'neutral':
                percentages += '-'

        try:
            window.write_event_value("New Emotion", {"emotion": result['dominant_emotion'],
                                                     "time": musicTime,
                                                     "percentages": percentages})
        except:
            'ignore'  # tk error, from tkinder library we're not using

        # ---------- Updates Counters ----------
        if result['dominant_emotion'] != 'Not Found':
            emotionsCounter[result['dominant_emotion']] += 1

        time.sleep(1)
        musicTime += 3

        if stop_main_emotions.is_set():
            print('STOP EMOTIONS')
            break
# main()
