Simple CamIO

Requirements: To run Simple CamIO, one needs to set up several things. 

-Firstly, a printed map with Aruco markers in the corners, as in map_with_aruco.png. The markers should be from the DICT_4X4_50 dictionary, with ids 0-3, placed in positions as specified in map_parameters.py.  

-Secondly, a digital version of the map that represents zones with a different index, as in zone_map.png, and this filename should be specified in the argument as input1. The precise resolution of the map should be specified in the variable pixels_per_cm_obj (in units of pixels per cm), such that it matches the printed version. 

-The camera should be calibrated and the calibration parameters should be specified by the variables focal_length_x, focal_length_y, camera_center_x, camera_center_y. Distortion parameters are also settable via the variable distortion (although these can safely be left to zero, assuming little distortion). 

-A pointer needs to be printed and cut out, using the pattern found in marker_pointer.png (printed at 300 dpi, such that the Aruco marker is exactly 3cm).

-Sound files, as named in the sound_dict dictionary in map_parameters.py, should be placed in the sound_files folder. The dictionary maps the zone index (from the zone map) to the sound file.

-Setting the camera: if an external webcam is to be used, set use_external_cam to 1, otherwise use 0 for the laptop's internal camera (or if you get a camera out of bounds error).

-Python 3.8 installed with opencv, numpy, scipy, and playsound libraries (most of which can be installed through Anaconda, except playsound which needs to be installed via pip)

For best performance, we recommend the camera sit above the map to get a fronto-parallel view as much as possible. The camera should have an unobstructed view of the 4 Aruco markers on the map, and the pointer should be held such that the camera can clearly view the marker.

To run, simply run the simple_camio.py script 
