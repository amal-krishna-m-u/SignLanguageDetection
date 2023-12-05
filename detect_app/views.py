# detection/views.py
from django.shortcuts import render
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import threading

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model_path = "path/to/your/model/Sign_language.h5"
model = load_model(model_path)
actions = np.array(["Hello", "Thanks", "ILoveYou"])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
threshold = 0.6

sequence = []
sentence = []
predictions = []

capture = cv2.VideoCapture(0)

def detection_thread():
    global sequence, sentence, predictions
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while capture.isOpened():
            return_, frame = capture.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 1:
                    sentence = sentence[-1:]

                image = prob_visualisation(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                " ".join(sentence),
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Feed", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 121), thickness=2, circle_radius=2),
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(8245, 66, 230), thickness=2, circle_radius=2),
    )

def extract_keypoints(results):
    pose = (
        np.array(
            [
                [result.x, result.y, result.z, result.visibility]
                for result in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(132)
    )

    face = (
        np.array(
            [
                [result.x, result.y, result.z]
                for result in results.face_landmarks.landmark
            ]
        ).flatten()
        if results.face_landmarks
        else np.zeros(1404)
    )

    left_hand = (
        np.array(
            [
                [result.x, result.y, result.z]
                for result in results.left_hand_landmarks.landmark
            ]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    right_hand = (
        np.array(
            [
                [result.x, result.y, result.z]
                for result in results.right_hand_landmarks.landmark
            ]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, face, left_hand, right_hand])

def prob_visualisation(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame

def detection_view(request):
    threading.Thread(target=detection_thread).start()
    return render(request, "detection/detection.html")
