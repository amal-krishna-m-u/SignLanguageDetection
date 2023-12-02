import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import argparse

# ==================================== Helper function ====================================
mp_holistic = mp.solutions.holistic         # Holistic model
mp_drawing = mp.solutions.drawing_utils     #Drawing utilites

def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Colour Conversion to RGB
    image.flags.writeable = False                       # Image is no longer writeable

    results = model.process(image)                      # Making prediction 

    image.flags.writeable = True                         # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts back to BGR for video output

    return image, results

def draw_landmarks(image, results): # Draws the landmarks and the connections on the image.
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),  # dot colour
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) # line colour

    # Draw pose connection     
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  # dot colour
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) # line colour

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),  # dot colour
                              mp_drawing.DrawingSpec(color=(121,44,121), thickness=2, circle_radius=2)) # line colour  

    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),  # dot colour
                              mp_drawing.DrawingSpec(color=(8245,66,230), thickness=2, circle_radius=2)) # line colour 
    
def extract_keypoints(results):
    pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten() \
       if results.pose_landmarks else np.zeros(132) # z value represents the depth or distance of a particular landmark point in relation to the camera or reference point.
    
    face = np.array([[result.x, result.y, result.z] for result in results.face_landmarks.landmark]).flatten() \
            if results.face_landmarks else np.zeros(1404)
    
    # if no left hand is detected we will just insert zero array with shape (21*3) there are 21 points for hand and 3 coordinates
    left_hand = np.array([[result.x, result.y, result.z] for result in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21*3)

    right_hand = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, left_hand, right_hand])

def prob_visualisation(res,actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1) #-1 means fill in the rectangle
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

# ==================================== Model ====================================
# Setup command line arguments
parser = argparse.ArgumentParser(description='Run the SignTalk real-time sign language interpreter.')
parser.add_argument('model_path', type=str, help='Path to the .h5 model file.')
args = parser.parse_args()
# Load the model from the provided path
model = load_model(args.model_path)

# Actions that we try to detect
actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
colors = [(245,117,16),(117,245,16),(16,117,245)] #different colours for different actions

# Detection Variables
sequence = [] # concat the np frames (30 frames) and append to sequence as list to be processed by the model
sentence = []
predictions = []
threshold = 0.6 

capture = cv2.VideoCapture(0) #grab camera device default -- usually webcam -- device 0 is iphone and device 1 is macbook camera

#Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():     # if camera is available

        # Read frame
        return_, frame = capture.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw results landmarks connections on image
        draw_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints) #insert keypoints to the front of the sequence list
        sequence = sequence[-30:] #grab the last 30 frames

        if len(sequence) == 30: # only run predicts when the number of frames appended is 30
            res = model.predict(np.expand_dims(sequence, axis=0))[0] #we expand the dim to 'fake' a batch size dim, NxCxHxW expand in N dim
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualise the logic
            if np.unique(predictions[-10:])[0]==np.argmax(res):     # grab the last 10 predictions and checking if ten predictions are similar to each other and the current prediction
                if res[np.argmax(res)] > threshold:                 #checking the highest prediction probability > threshold
                    if len(sentence) > 0:                           #check if actions have already been predicted
                        if actions[np.argmax(res)] != sentence[-1]: # to check if the current action differs with the previous sentence
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:] #grab only the last 5 sentences

            # Visualise the probabilities
            image = prob_visualisation(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        #Show the frame with the landmarks connections from image
        cv2.imshow("Feed", image)

        #break gracefully from the lopp
        if cv2.waitKey(1) & 0xFF == ord('q'): #& 0xFF operation ensures that the key code is compared correctly with the ASCII code of 'q' (ord('q')).
            break

    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)