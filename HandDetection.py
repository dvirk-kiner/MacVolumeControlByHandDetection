import cv2
import mediapipe as mp
import operator
import math
import osascript

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands


class HandDetection:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(static_image_mode, max_num_hands,
                                         min_detection_confidence, min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, image, isDrawing=True):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        self.results = self.hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if isDrawing:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def findPosition(self, image, isDrawing=True):
        handList = []
        if self.results is not None:
            if self.results.multi_hand_landmarks:
                leftHand = self.results.multi_hand_landmarks[0]
                for idH, lmH in enumerate(leftHand.landmark):
                    if(idH == 4 or idH == 8):
                        h, w, c = image.shape
                        cx, cy = int(lmH.x * w), int(lmH.y * h)
                        handList.append((cx, cy))
                        if isDrawing:
                            cv2.circle(
                                image, (cx, cy), 12, (0, 255, 255), cv2.FILLED)
        return image, handList


def normelizeLength(currentLen, a=0, b=100, minGiven=7, maxGiven=180):
    finalNum = int((((b-a)*(currentLen-minGiven))/(maxGiven-minGiven))+a)
    if finalNum < 0:
        finalNum = 0
    if finalNum > 100:
        finalNum = 100
    return finalNum


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    detector = HandDetection()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        image = detector.findHands(image)
        image, handList = detector.findPosition(image)
        if len(handList) > 1:
            cv2.line(image, handList[0], handList[1], (255, 0, 255), 1)
            cx, cy = tuple(map(operator.floordiv, tuple(
                map(operator.add, handList[0], handList[1])), (2, 2)))
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), 5)

            length = math.hypot(
                handList[0][0]-handList[1][0], handList[0][1]-handList[1][1])
            # A number between 0 to 100
            normelizeLen = normelizeLength(length)
            osascript.run("set volume output volume "+str(normelizeLen))
            _, out, _ = osascript.run(
                "output volume of (get volume settings)")
            print(str(normelizeLen)+"---"+str(out))

        # # Flip the image horizontally for a later selfie-view display, and convert
        # # the BGR image to RGB.
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # # To improve performance, optionally mark the image as not writeable to
        # # pass by reference.
        # image.flags.writeable = False
        # results = hands.process(image)

        # # Draw the hand annotations on the image.
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Volume controlled by Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


main()
