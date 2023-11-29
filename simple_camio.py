import os
import cv2 as cv
import datetime
import time
import numpy as np
import pickle
import json
import argparse
import pyglet.media
from scipy import stats


# The ModelDetector class is responsible for detecting the Aruco markers in
# the image that make up the model, and determining the pose of the model in
# the scene.
class ModelDetector:
    def __init__(self, model, intrinsic_matrix):
        # Parse the Aruco markers placement positions from the parameter file into a numpy array, and get the associated ids
        self.obj, self.list_of_ids = parse_aruco_codes(model['positioningData']['arucoCodes'])
        # Define aruco marker dictionary and parameters object to include subpixel resolution
        self.aruco_dict_scene = cv.aruco.Dictionary_get(
            get_aruco_dict_id_from_string(model['positioningData']['arucoType']))
        self.arucoParams = cv.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.intrinsic_matrix = intrinsic_matrix

    def detect(self, frame):
        # Detect the markers in the frame
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame, self.aruco_dict_scene, parameters=self.arucoParams)
        scene, use_index = sort_corners_by_id(corners, ids, self.list_of_ids)
        if ids is None or not any(use_index):
            print("No markers found.")
            return False, None, None

        # Run solvePnP using the markers that have been observed to determine the pose
        retval, rvec, tvec = cv.solvePnP(self.obj[use_index, :], scene[use_index, :], self.intrinsic_matrix, None)
        return retval, rvec, tvec


# The PoseDetector class determines the pose of the pointer, and returns the
# position of the tip in the coordinate system of the model.
class PoseDetector:
    def __init__(self, stylus, intrinsic_matrix):
        # Parse the Aruco markers placement positions from the parameter file into a numpy array, and get the associated ids
        self.obj, self.list_of_ids = parse_aruco_codes(stylus['positioningData']['arucoCodes'])
        self.aruco_dict = cv.aruco.Dictionary_get(get_aruco_dict_id_from_string(stylus['positioningData']['arucoType']))
        self.arucoParams = cv.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.intrinsic_matrix = intrinsic_matrix

    def detect(self, frame, rvec_model, tvec_model):
        corners, ids, _ = cv.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.arucoParams)
        scene, use_index = sort_corners_by_id(corners, ids, self.list_of_ids)
        if ids is None or not any(use_index):
            print("No pointer detected.")
            return None
        retval, rvec_arcuo, tvec_aruco = cv.solvePnP(self.obj, scene, self.intrinsic_matrix, None)

        # Get pointer location in coordinates of the aruco markers
        point_of_interest = reverse_project(tvec_aruco, rvec_model, tvec_model)
        return point_of_interest


# The InteractionPolicy class takes the position and determines where on the
# map it is, finding the color of the zone, if any, which is decoded into
# zone ID number. This zone ID number is filtered through a ring buffer that
# returns the mode. If the position is near enough to the plane (within 2cm)
# then the zone ID number is returned.
class InteractionPolicy:
    def __init__(self, model):
        self.model = model
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)
        self.zone_filter_size = 10
        self.zone_filter = -1 * np.ones(self.zone_filter_size, dtype=int)
        self.zone_filter_cnt = 0

    def push_gesture(self, position):
        zone_color = get_zone(position, self.image_map_color, self.model['pixels_per_cm'])
        self.zone_filter[self.zone_filter_cnt] = get_dict_idx_from_color(self.model['hotspots'], zone_color)
        self.zone_filter_cnt = (self.zone_filter_cnt + 1) % self.zone_filter_size
        zone = stats.mode(self.zone_filter).mode[0]
        if np.abs(position[2]) < 2.0:
            return zone
        else:
            return -1


class CamIOPlayer:
    def __init__(self, model):
        self.model = model
        self.prev_zone_name = ''
        self.start_time = time.time()
        self.sound_files = []
        for hotspot in self.model['hotspots']:
            if os.path.exists(hotspot['audioDescription']):
                self.sound_files.append(pyglet.media.load(hotspot['audioDescription'], streaming=False))
            else:
                print("warning. file not found:" + hotspot['audioDescription'])

    def convey(self, zone):
        zone_name = self.model['hotspots'][zone]['textDescription']
        if self.prev_zone_name != zone_name:
            if time.time() - self.start_time > 0.5:
                sound = self.sound_files[zone]
                try:
                    sound.play()
                except(BaseException):
                    print("Exception raised. Cannot play sound. Please restart the application.")
                self.start_time = time.time()
                self.prev_zone_name = zone_name


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append((dev_port, h, w))
            else:
                print("Port %s for camera ( %s x %s) is present but does not read." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


# Function to load camera intrinsic parameters from a JSON file
def load_camera_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            cam_params = json.load(f)
            print("loaded camera parameters from file.")
    else:
        print("No camera parameters file found. Please run simple_calibration.py script.")
        exit(0)
    intrinsic_matrix = np.array([[cam_params['focal_length_x'], 0.0, cam_params['camera_center_x']],
                                 [0.00000000e+00, cam_params['focal_length_y'], cam_params['camera_center_y']],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    return intrinsic_matrix


# Function to load map parameters from a JSON file
def load_map_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            map_params = json.load(f)
            print("loaded map parameters from file.")
    else:
        print("No map parameters file found.")
        exit(0)
    return map_params['model']


# Function to load map parameters from a JSON file
def load_stylus_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            stylus_params = json.load(f)
            print("loaded stylus parameters from file.")
    else:
        print("No stylus parameters file found.")
        exit(0)
    return stylus_params['stylus']


# Function to sort corners by id based on the order specified in the id_list,
# such that the scene array matches the obj array in terms of aruco marker ids
# and the appropriate corners.
def sort_corners_by_id(corners, ids, id_list):
    use_index = np.zeros(len(id_list)*4, dtype=bool)
    scene = np.empty((len(id_list) * 4, 2), dtype=np.float32)
    for i in range(len(corners)):
        id = ids[i, 0]
        if id in id_list:
            corner_num = id_list.index(id)
            for j in range(4):
                use_index[4 * corner_num + j] = True
                scene[4 * corner_num + j, :] = corners[i][0][j]
    return scene, use_index


# Function to reverse the projection of a point given a rvec and tvec
def reverse_project(point, rvec, tvec):
    poi = np.matrix(point)
    R, _ = cv.Rodrigues(rvec)
    R_mat = np.matrix(R)
    R_inv = np.linalg.inv(R_mat)
    T = np.matrix(tvec)
    return np.array(R_inv * (poi - T))


# Function to create 3D points from 2D pixels on a sheet of paper
def get_3d_points_from_pixels(obj_pts, pixels_per_cm):
    pts_3d = np.empty((len(obj_pts), 3), dtype=np.float32)
    for i in range(len(obj_pts)):
        pts_3d[i, 0] = obj_pts[i, 0] / pixels_per_cm
        pts_3d[i, 1] = obj_pts[i, 1] / pixels_per_cm
        pts_3d[i, 2] = 0
    return pts_3d


# Draws the axes on the image
def drawAxes(img, imgpts):
    imgpts = imgpts.astype(int)
    corner = tuple(imgpts[3].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


# Draws axes and projects the 3D points onto the image
def annotate_image(img_scene_color, obj, rvec, tvec, intrinsic_matrix):
    # Draws the 3D points on the image
    backprojection_pts, other = cv.projectPoints(obj, rvec, tvec, intrinsic_matrix, None)
    for idx, pts in enumerate(backprojection_pts):
        cv.circle(img_scene_color, (int(pts[0, 0]), int(pts[0, 1])), 4, (255, 255, 255), 2)
        cv.line(img_scene_color, (int(pts[0, 0] - 1), int(pts[0, 1])), (int(pts[0, 0]) + 1, int(pts[0, 1])),
                (255, 0, 0), 1)
        cv.line(img_scene_color, (int(pts[0, 0]), int(pts[0, 1]) - 1), (int(pts[0, 0]), int(pts[0, 1]) + 1),
                (255, 0, 0), 1)
    # Draw axes on the image
    axis = np.float32([[6, 0, 0], [0, 6, 0], [0, 0, -6], [0, 0, 0]]).reshape(-1, 3)
    axis_pts, other = cv.projectPoints(axis, rvec, tvec, intrinsic_matrix, None)
    img_scene_color = drawAxes(img_scene_color, axis_pts)
    return img_scene_color


# Retrieves the zone of the point of interest on the map
def get_zone(point_of_interest, img_map, pixels_per_cm):
    x = int(point_of_interest[0] * pixels_per_cm)
    y = int(point_of_interest[1] * pixels_per_cm)
    if 0 <= x < img_map.shape[1] and 0 <= y < img_map.shape[0]:
        return img_map[y, x]
    else:
        return 0


def match_color(rgb_color_list, bgr_color_np_array):
    if np.array_equal(np.array(rgb_color_list[::-1]), np.squeeze(bgr_color_np_array)):
        return True
    else:
        return False


# Returns the dictionary in the list of dictionaries that matches the color given
def get_dict_from_color(list_of_dicts, color):
    for dictionary in list_of_dicts:
        if (dictionary['color'] == color):
            return dictionary
    return None


# Returns the index of the dictionary in the list of dictionaries that matches the color given
def get_dict_idx_from_color(list_of_dicts, color):
    for i in range(len(list_of_dicts)):
        dictionary = list_of_dicts[i]
        if match_color(dictionary['color'], color):
            return i
    return -1


# Parses the list of aruco codes and returns the 2D points and ids
def parse_aruco_codes(list_of_aruco_codes):
    obj_array = np.empty((len(list_of_aruco_codes)*4,3), dtype=np.float32)
    ids = []
    for cnt, aruco_code in enumerate(list_of_aruco_codes):
        for i in range(4):
            obj_array[cnt*4+i,:] = aruco_code['position'][i]
        ids.append(aruco_code['id'])
    return obj_array, ids


# Returns the dictionary code for the given string
def get_aruco_dict_id_from_string(aruco_dict_string):
    if aruco_dict_string == "DICT_4X4_50":
        return cv.aruco.DICT_4X4_50
    elif aruco_dict_string == "DICT_4X4_100":
        return cv.aruco.DICT_4X4_100
    elif aruco_dict_string == "DICT_4X4_250":
        return cv.aruco.DICT_4X4_250
    elif aruco_dict_string == "DICT_4X4_1000":
        return cv.aruco.DICT_4X4_1000
    elif aruco_dict_string == "DICT_5X5_50":
        return cv.aruco.DICT_5X5_50
    elif aruco_dict_string == "DICT_5X5_100":
        return cv.aruco.DICT_5X5_100
    elif aruco_dict_string == "DICT_5X5_250":
        return cv.aruco.DICT_5X5_250

available_ports, working_ports, non_working_ports = list_ports()
print(f"working ports: {working_ports}")
# ========================================
USE_EXTERNAL_CAM = 0
#========================================

parser = argparse.ArgumentParser(description='Code for CamIO.')
parser.add_argument('--input1', help='Path to input zone image.', default='UkraineMap.json')
args = parser.parse_args()

# Load map and camera parameters
model = load_map_parameters(args.input1)
intrinsic_matrix = load_camera_parameters('camera_parameters.json')
stylus = load_stylus_parameters('TeardropStylus.json')

model_detector = ModelDetector(model, intrinsic_matrix)
pose_detector = PoseDetector(stylus, intrinsic_matrix)
interact = InteractionPolicy(model)
camio_player = CamIOPlayer(model)

cap = cv.VideoCapture(USE_EXTERNAL_CAM)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080) #set camera image height
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920) #set camera image width
cap.set(cv.CAP_PROP_FOCUS,0)
loop_has_run = False

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No camera image returned.")
        break
    if loop_has_run:
        cv.imshow('image reprojection', img_scene_color)
        waitkey = cv.waitKey(1)
        if waitkey == 27 or waitkey == ord('q'):
            print('Escape.')
            cap.release()
            cv.destroyAllWindows()
            break

    img_scene_color = frame
    loop_has_run = True

    # load images grayscale
    img_scene = cv.cvtColor(img_scene_color, cv.COLOR_BGR2GRAY)
    # Detect aruco markers for map in image
    retval, rvec, tvec = model_detector.detect(img_scene)

    # If no  markers found, continue to next iteration
    if not retval:
        print("No markers found.")
        continue

    # Annotate image with 3D points and axes
    img_scene_color = annotate_image(img_scene_color, model_detector.obj, rvec, tvec, intrinsic_matrix)

    # Detect aruco marker for pointer in image
    point_of_interest = pose_detector.detect(img_scene, rvec, tvec)

    # If no pointer is detected, move on to the next frame
    if point_of_interest is None:
        print("No pointer detected.")
        continue

    # Determine zone from point of interest
    zone_id = interact.push_gesture(point_of_interest)

    # If the zone id is valid, play the sound for the zone
    if zone_id > -1:
        camio_player.convey(zone_id)
