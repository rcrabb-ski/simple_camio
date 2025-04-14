import numpy as np
import cv2 as cv
import time
import mediapipe as mp

class PoseDetectorMP3D:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = self.hands.process(image)
        coors = np.zeros((4,3), dtype=float)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for k in [1, 2, 3, 4]:  # joints in thumb
                    coors[k - 1, 0], coors[k - 1, 1], coors[k - 1, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_thumb = self.ratio(coors)

                for k in [5, 6, 7, 8]:  # joints in index finger
                    coors[k - 5, 0], coors[k - 5, 1], coors[k - 5, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_index = self.ratio(coors)
                a = coors[0,:]
                ab = coors[3,:] - coors[0,:]

                for k in [9, 10, 11, 12]:  # joints in middle finger
                    coors[k - 9, 0], coors[k - 9, 1], coors[k - 9, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_middle = self.ratio(coors)
                is_pointing = True
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                for k in [13, 14, 15, 16]:  # joints in ring finger
                    coors[k - 13, 0], coors[k - 13, 1], coors[k - 13, 2] = hand_landmarks.landmark[k].x, \
                                                                           hand_landmarks.landmark[k].y, \
                                                                           hand_landmarks.landmark[k].z
                ratio_ring = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                for k in [17, 18, 19, 20]:  # joints in little finger
                    coors[k - 17, 0], coors[k - 17, 1], coors[k - 17, 2] = hand_landmarks.landmark[k].x, \
                                                                           hand_landmarks.landmark[k].y, \
                                                                           hand_landmarks.landmark[k].z
                ratio_little = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                print(ratio_thumb, ratio_index, ratio_middle, ratio_ring, ratio_little)
                overall = ratio_index - ((ratio_middle + ratio_ring + ratio_little) / 3)
                print('overall evidence for index pointing:', overall)

                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                position = np.array(
                    [hand_landmarks.landmark[8].x * image.shape[1], hand_landmarks.landmark[8].y * image.shape[0]])

                if overall > 0.1 or is_pointing:
                    print(hand_landmarks.landmark[8])
                    return position, "pointing", image
                else:
                    return position, "moving", image
        return None, None, image


    def ratio(self, coors):  # ratio is 1 if points are collinear, lower otherwise (minimum is 0)
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])

        return d / (a + b + c)


class TouchModelDetector:
    def __init__(self, model, intrinsic_matrix):
        self.model = model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.rvec = None
        self.tvec = None
        self.requires_pnp = True
        self.intrinsic_matrix = intrinsic_matrix
        self.positions = []
        self.obj_positions = np.array(model['obj_positions_3d'], dtype=np.float32)
        self.timer = time.time()

    def detect(self, frame):
        # If we have already computed the coordinate transform then simply return it
        if not self.requires_pnp:
            return True, self.rvec, self.tvec
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        coors = np.zeros((4, 3), dtype=float)
        position = None
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for k in [1, 2, 3, 4]:  # joints in thumb
                    coors[k - 1, 0], coors[k - 1, 1], coors[k - 1, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_thumb = self.ratio(coors)

                for k in [5, 6, 7, 8]:  # joints in index finger
                    coors[k - 5, 0], coors[k - 5, 1], coors[k - 5, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_index = self.ratio(coors)
                a = coors[0,:]
                ab = coors[3,:] - coors[0,:]

                for k in [9, 10, 11, 12]:  # joints in middle finger
                    coors[k - 9, 0], coors[k - 9, 1], coors[k - 9, 2] = hand_landmarks.landmark[k].x, \
                                                                        hand_landmarks.landmark[k].y, \
                                                                        hand_landmarks.landmark[k].z
                ratio_middle = self.ratio(coors)
                is_pointing = True
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                for k in [13, 14, 15, 16]:  # joints in ring finger
                    coors[k - 13, 0], coors[k - 13, 1], coors[k - 13, 2] = hand_landmarks.landmark[k].x, \
                                                                           hand_landmarks.landmark[k].y, \
                                                                           hand_landmarks.landmark[k].z
                ratio_ring = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                for k in [17, 18, 19, 20]:  # joints in little finger
                    coors[k - 17, 0], coors[k - 17, 1], coors[k - 17, 2] = hand_landmarks.landmark[k].x, \
                                                                           hand_landmarks.landmark[k].y, \
                                                                           hand_landmarks.landmark[k].z
                ratio_little = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                print(ratio_thumb, ratio_index, ratio_middle, ratio_ring, ratio_little)
                overall = ratio_index - ((ratio_middle + ratio_ring + ratio_little) / 3)
                print('overall evidence for index pointing:', overall)

                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                if overall > 0.1 or is_pointing:
                    position = np.array([hand_landmarks.landmark[8].x * image.shape[1], hand_landmarks.landmark[8].y * image.shape[0]])
        for pos in self.positions:
            cv.circle(frame, (int(pos[0]), int(pos[1])), 3, (0, 255, 0), -1)
        cv.imshow('MediaPipe Hands', frame)
        waitkey = cv.waitKey(1)
        if waitkey == ord(' ') and position is not None and time.time() - self.timer > 3:
            self.timer = time.time()
            self.positions.append(position)
        if len(self.positions) < len(self.obj_positions):
            return False, None, None
        scene_inliers = np.array(self.positions, dtype=np.float32)
        # Run PnP to get rotation and translation vectors
        retval, self.rvec, self.tvec = cv.solvePnP(self.obj_positions, scene_inliers, self.intrinsic_matrix, None)
        self.requires_pnp = not retval
        return retval, self.rvec, self.tvec

    def ratio(self, coors):  # ratio is 1 if points are collinear, lower otherwise (minimum is 0)
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])

        return d / (a + b + c)