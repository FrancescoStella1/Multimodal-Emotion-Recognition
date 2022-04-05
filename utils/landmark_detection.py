import requests, sys
from os import path
from enum import Enum

import cv2
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import mediapipe as mp

from utils import load_store
from utils.video_utils import VideoManager


URL_LBFMODEL = 'https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml'


LOWER_LANDMARK, UPPER_LANDMARK = 17, 68     # from right_eye ID to mouth ID
SHOW_FRAMES = False


class LDMK_TYPE(Enum):
    """
    Enum containing indexes of relevant landmarks aimed at the construction
    of the facial features vector. 
    """
    LIPS = {"upper": 34, "lower": 40}
    EYES = {"left": [(20, 24), (21, 23)], "right": [(26, 30), (27, 29)]}
    MOUTH_ANGLE = {"left": [31, 40], "right": [37, 40]}
    EYEBROWS = {"left": np.linspace(0, 4, 5, dtype=np.int), "right": np.linspace(5, 9, 5, dtype=np.int)}


class LDMK_TYPE_MP(Enum):
    """
    Enum containing indexes of relevant landmarks for Mediapipe model
    aimed at the construction of the facial features vector. 
    """
    LIPS = {"upper": 0, "lower": 17}
    EYES = {"left": [(160, 144), (158, 153)], "right": [(385, 380), (387, 373)]}
    MOUTH_ANGLE = {"left": [61, 17], "right": [291, 17]}
    EYEBROWS = {"left": [46, 53, 52, 65, 55], "right": [285, 295, 282, 283, 276]}


def compute_features(points, mapping):
    r"""
    Computes facial features starting from detected landmarks.

    Parameters
    ----------
    points: array_like
        Array of (x,y)-couples representing the coordinates of the detected landmarks.
    mapping: dict
        Dict {face part: points indexes} containing the mapping of the relevant face parts (such as eyes, lips, etc.)
        to the indexes of the corresponding detected landmarks. These are the landmarks actually used for feature extraction.

    Returns
    -------
    features: array_like
        Array of the facial features.
    """
    features = np.array([])
    # Lips features
    upper_lip = mapping["upper lip"]
    lower_lip = mapping["lower lip"]
    x = points[upper_lip][0] - points[lower_lip][0]
    y = points[upper_lip][1] - points[lower_lip][1]
    dist = np.sqrt(x**2 + y**2)
    features = np.concatenate((features, np.array([dist])))

    # Eyes features
    l_eye = mapping["left eye"]
    r_eye = mapping["right eye"]

    # Here, points[coord][0] represents the x-coordinate of the landmark with index coord in points
    # while points[coord][1] is the y-coordinate of the same landmark.
    # Eye openings take into account only the y-coordinate of specific landmarks.
    l_eye_opening = 1/2*((points[l_eye[0][0]][1] - points[l_eye[0][1]][1]) + (points[l_eye[1][0]][1] - points[l_eye[1][1]][1]))
    features = np.concatenate((features, np.array([l_eye_opening])))
    r_eye_opening = 1/2*((points[r_eye[0][0]][1] - points[r_eye[0][1]][1]) + (points[r_eye[1][0]][1] - points[r_eye[1][1]][1]))
    features = np.concatenate((features, np.array([r_eye_opening])))

    # Mouth features
    l_mouth = mapping["left mouth"]
    r_mouth = mapping["right mouth"]
    l_mouth_tmp = (points[l_mouth[0]][1] - points[l_mouth[1]][1]) / (points[l_mouth[1]][0] - points[l_mouth[0]][0])
    l_mouth_ang = np.arctan(l_mouth_tmp)
    features = np.concatenate((features, np.array([l_mouth_ang])))
    r_mouth_tmp = (points[r_mouth[0]][1] - points[r_mouth[1]][1]) / (points[r_mouth[0]][0] - points[r_mouth[1]][0])
    r_mouth_ang = np.arctan(r_mouth_tmp)
    features = np.concatenate((features, np.array([r_mouth_ang])))

    # Eyebrow features
    l_eyebrow = mapping["left eyebrow"] #LDMK_TYPE.EYEBROWS.value["left"]
    r_eyebrow = mapping["right eyebrow"] #LDMK_TYPE.EYEBROWS.value["right"]

    l_eyebrow_fit = Polynomial.fit(np.array([points[x][0] for x in l_eyebrow]),
                                   np.array([points[x][1] for x in l_eyebrow]), 1)
    l_slope = np.array([l_eyebrow_fit.convert().coef[1]])
    features = np.concatenate((features, l_slope))

    r_eyebrow_fit = Polynomial.fit(np.array([points[x][0] for x in r_eyebrow]),
                                   np.array([points[x][1] for x in r_eyebrow]), 1)
    r_slope = np.array([r_eyebrow_fit.convert().coef[1]])
    features = np.concatenate((features, r_slope))

    return features


def get_features(filepath, face_model, landmarks_model, skip_frames=25):
    r"""
    Computes facial features by first extracting landmarks from filepath.

    Parameters
    ----------
    filepath: str
        Path to the video clip from which to extract features.
    face_model: object
        Cascade classifier used to detect faces.
    landmarks_model: object
        Local Binary Fitting (LBF) model used to extract face landmarks.
    skip_frames: int
        Number of frames to skip after each frame processing (default: 25)

    Returns
    -------
    features_vector: array_like
        Array of the extracted facial features.
    """
    extractor = VideoManager(filepath, skip_frames)
    frame = extractor.read_frame()
    features_vector = np.array([])
    mapping = {
        "upper lip": LDMK_TYPE.LIPS.value["upper"],
        "lower lip": LDMK_TYPE.LIPS.value["lower"],
        "left eye": LDMK_TYPE.EYES.value["left"],
        "right eye": LDMK_TYPE.EYES.value["right"],
        "left mouth": LDMK_TYPE.MOUTH_ANGLE.value["left"],
        "right mouth": LDMK_TYPE.MOUTH_ANGLE.value["right"],
        "left eyebrow": LDMK_TYPE.EYEBROWS.value["left"],
        "right eyebrow": LDMK_TYPE.EYEBROWS.value["right"]
    }
    while frame is not None:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)
        faces = face_model.detectMultiScale(grayscale, 1.1, 5, minSize=(190, 190))        # bounding boxes
        if len(faces)==0:
            continue
        _, landmarks = landmarks_model.fit(grayscale, faces)
        for x, y, w, h in faces:
            for ld in landmarks:
                if len(features_vector)==0:
                    features_vector = compute_features(ld[0][LOWER_LANDMARK:UPPER_LANDMARK], mapping)
                else:
                    features_vector = np.row_stack((features_vector,
                                                    compute_features(ld[0][LOWER_LANDMARK:UPPER_LANDMARK], mapping)))
                if SHOW_FRAMES:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    for x, y in ld[0][LOWER_LANDMARK:UPPER_LANDMARK]:
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 1)
        if SHOW_FRAMES:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                sys.exit(1)
        frame = extractor.read_frame()
    
    return features_vector


def get_features_mp(filepath, filename):
    r"""
    Computes facial features using the landmarks in the .npz file obtained through the Mediapipe FaceMesh model.

    Parameters
    ----------
    filepath: str
        Path to the .npz file containing the landmarks.
    filename: str
        Name of the .npz file containing the landmarks.

    Returns
    -------
    features_vector: array_like
        Array containing the facial features computed for each frame in the file.
    """
    try:
        landmarks = load_store.load_landmark_data(filepath, filename)
        features_vector = np.array([])
        mapping = {
            "upper lip": LDMK_TYPE_MP.LIPS.value["upper"],
            "lower lip": LDMK_TYPE_MP.LIPS.value["lower"],
            "left eye": LDMK_TYPE_MP.EYES.value["left"],
            "right eye": LDMK_TYPE_MP.EYES.value["right"],
            "left mouth": LDMK_TYPE_MP.MOUTH_ANGLE.value["left"],
            "right mouth": LDMK_TYPE_MP.MOUTH_ANGLE.value["right"],
            "left eyebrow": LDMK_TYPE_MP.EYEBROWS.value["left"],
            "right eyebrow": LDMK_TYPE_MP.EYEBROWS.value["right"]
        }
        for ldmk_array in landmarks:      # for each saved frame
            if len(features_vector) == 0:
                features_vector = compute_features(ldmk_array, mapping)
            else:
                features_vector = np.row_stack((features_vector, compute_features(ldmk_array, mapping)))

        return features_vector

    except Exception as e:
        print("Exception while extracting features with mediapipe-facemesh from: ", filepath, "\n", repr(e),
              file=sys.stderr)


def get_landmarks_mp(filepath, skip_frames=25):
    r"""
    Computes facial landmarks using the video in filepath.

    Parameters
    ----------
    filepath: str
        Path to the video.
    skip_frames: int
        Number of frames to skip after each frame processing (default: 25).

    Returns
    -------
    landmarks_arr: array_like
        Array containing x,y-coordinates for each landmark in each extracted frame.
    """
    try:
        reader = VideoManager(filepath, skip_frames)
        print("\nExtracting landmarks from: ", filepath)
        mp_drawing = mp.solutions.drawing_utils
        #mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        #drawing_spec = mp_drawing.DrawingSpec() #thickness=0.03, circle_radius=0.03)
        landmarks_arr = np.array([])
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            frame = reader.read_frame()
            while frame is not None:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = np.array([])
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        if SHOW_FRAMES:
                            mp_drawing.draw_landmarks(frame, landmark_list=face_landmarks)
                        face_landmarks = face_landmarks.ListFields()[0][1]
                        for ld in face_landmarks:
                            ld = np.array([ld.x * frame.shape[1], ld.y * frame.shape[0]])         # get x,y coordinates
                            ld = np.reshape(ld, (1, 2))
                            if len(landmarks)==0:
                                landmarks = ld
                            else:
                                landmarks = np.row_stack((landmarks, ld))
                        
                        landmarks = np.reshape(landmarks, (1, landmarks.shape[0], landmarks.shape[1]))
                        if len(landmarks_arr) == 0:
                            landmarks_arr = landmarks
                        else:
                            landmarks_arr = np.row_stack((landmarks_arr, landmarks))

                if SHOW_FRAMES:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        sys.exit(1)
                frame = reader.read_frame()
        return landmarks_arr

    except Exception as e:
        print("Exception while extracting landmarks with mediapipe-facemesh from: ", filepath, "\n",
              repr(e), file=sys.stderr)


def load_models():
    """Loads and returns face and landmarks models."""
    face_model = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    landmarks_model = cv2.face.createFacemarkLBF()
    if not(path.exists('./models/lbfmodel.yaml')):
        download = requests.get(URL_LBFMODEL)
        open('./models/lbfmodel.yaml', 'wb').write(download.content)
    landmarks_model.loadModel("./models/lbfmodel.yaml")
    return face_model, landmarks_model