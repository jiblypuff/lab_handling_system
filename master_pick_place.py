# import the opencv library
import cv2
# import the 
from pydexarm import Dexarm
# import math
import math
import time

z_search = 110

# define a dexarm object to control the Dexarm
dexarm = Dexarm("COM3")
dexarm.go_home()
curr_pos = dexarm.get_current_position()
dexarm.move_to(curr_pos[0], curr_pos[1], z_search)
dexarm.soft_gripper_nature()

# define a video capture object
vid = cv2.VideoCapture(2, cv2.CAP_DSHOW)

state = 1
# represents the center of the object to be picked and placed
center_X = 0
center_Y = 0
# represents where the object needs to be located in the frame to be picked properly
goal_x = 322
goal_y = 405
# represents the height where the air picker will be just above the table surface
z_table = -52
# represents the height at which the arm will traverse across the table
z_trav = 50

# search rad
search_rad = 300

error_bound = 8
object_found = False
count = 0

x_lower_lim = -395
x_upper_lim = 395
y_lower_lim = 0
y_upper_lim = 395

# State Guide:
# State 1: Search for object in image
# State 2: Move Dexarm to proper location
# State 3: Pick using Dexarm
# State 4: Move object to proper location and drop
# State 5: Return to search zone

def check_arm_pos(x_coord, y_coord):
    if ((x_coord > goal_x - error_bound) and (x_coord < goal_x + error_bound)):
        if ((y_coord > goal_y - error_bound) and (y_coord < goal_y + error_bound)):
            return True
    return False

def move_radially(dist):
    print("Distance: " + str(dist))
    curr_radial_pos = dexarm.get_current_position()
    curr_radial_x = curr_radial_pos[0]
    curr_radial_y = curr_radial_pos[1]

    arm_angle = math.atan(curr_radial_y / (curr_radial_x + 0.000001))

    if (dist > 0):
        dist = 3
    else:
        dist = -3

    radial_cont_x = int(dist * math.cos(arm_angle))
    radial_cont_y = int(dist * math.sin(arm_angle))

    dexarm.fast_move_to(curr_radial_x + radial_cont_x, curr_radial_y + radial_cont_y, curr_radial_pos[2])

def rotate_angle(theta):
    curr_pos = dexarm.get_current_position()
    curr_x = curr_pos[0]
    curr_y = curr_pos[1]

    print("Inside Rotate Angle")

    curr_rad = math.sqrt(curr_x * curr_x + curr_y * curr_y)
    curr_theta = math.atan(curr_y / (curr_x + 0.000001))
    if (curr_theta < 0):
        curr_theta += math.pi
    
    if (theta > 0):
        theta = math.pi / 150
    else:
        theta = -math.pi / 150
    new_theta = curr_theta + theta

    print("Current Angle: " + str(curr_theta))
    print("Current Radius: " + str(curr_rad))
    print("New Angle: " + str(new_theta))

    new_x = int(curr_rad * math.cos(new_theta))
    print("New x: " + str(new_x))
    new_y = int(curr_rad * math.sin(new_theta))
    print("New y: " + str(new_y))

    dexarm.fast_move_to(new_x, new_y, curr_pos[2])

def valid_location(drop_x, drop_y):
    drop_x = int(drop_x)
    drop_y = int(drop_y)
    radius = math.sqrt(drop_x * drop_x + drop_y * drop_y)
    if (radius < 200 or radius > 395):
        return False
    else:
        return True
    

while(True):

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
    elif state == 1:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        print("Object found: " + str(object_found))

        # print('Original Dimensions: ', frame.shape)
        
        scale_percent = 100
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        frame = cv2.medianBlur(frame, 11)
        
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()

        areas = []
        centers = []

        for c in contours:
            # find the area of the contour
            areas.append(cv2.contourArea(c))
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + 1e-5))
            cY = int(M["m01"] / (M["m00"] + 1e-5))

            centers.append((cX, cY))

        all_conts = {}

        for i in range(len(areas)):
            all_conts[i] = {'area' : areas[i], 'center' : centers[i]}

        sorted_conts = sorted(all_conts.items(), key = lambda x:x[1]["area"], reverse = True)


        area_num = 0

        if(len(areas) != 0):
            for i in range(len(areas)):
                if(sorted_conts[i][1]['area'] * 100 > sorted_conts[0][1]['area']
                   and sorted_conts[i][1]['area'] * 2 < sorted_conts[0][1]['area']):
                    center_X = sorted_conts[i][1]['center'][0]
                    center_Y = sorted_conts[i][1]['center'][1]
                    cv2.drawContours(frame_copy, contours, sorted_conts[i][0], (0, 255, 0), 2, cv2.LINE_AA)
                    area_num = area_num + 1

        cv2.imshow("Contour Drawing", frame_copy)
        print("CenterX: " + str(center_X))
        print("CenterY: " + str(center_Y))
        print(dexarm.get_current_position())
        
        if (area_num > 0):
            object_found = True
            state = 2
        elif (not object_found):
            rotate_angle(-1)
            time.sleep(0.25)
        elif (count > 30):
            object_found = False
            count = 0
        else:   
            count += 1
        
    elif state == 2:
        if (check_arm_pos(center_X, center_Y)):
            state = 3
            print("Arm in correct position")
        else:
            x_error = goal_x - center_X
            y_error = goal_y - center_Y
            print("Y error: " + str(y_error))

            if (abs(x_error) > error_bound):
                rotate_angle(x_error)
                if (abs(x_error) < 2 * error_bound):
                    time.sleep(0.4)
                else:
                    time.sleep(0.4)
            if (abs(y_error) > error_bound):
                move_radially(y_error)
                if (abs(y_error) < 2 * error_bound):
                    time.sleep(0.5)
                else:
                    time.sleep(0.5)
            
            state = 1
    
    # Picks up the object and moves it to the drop zone
    elif state == 3:
        curr_pos = dexarm.get_current_position()
        dexarm.soft_gripper_place()
        dexarm.fast_move_to(curr_pos[0], curr_pos[1], z_table)
        dexarm.soft_gripper_pick()
        dexarm.fast_move_to(curr_pos[0], curr_pos[1], z_trav)
        
        state = 4

    # Moves the object from the original search zone to the drop zone
    elif state == 4:
        drop_x = input("Enter x location to drop at: ")
        drop_y = input("Enter y location to drop at: ")
        if(valid_location(drop_x, drop_y)):
            dexarm.fast_move_to(drop_x, drop_y, z_trav)
            dexarm.fast_move_to(drop_x, drop_y, z_table)
            dexarm.soft_gripper_place()
            dexarm.fast_move_to(drop_x, drop_y, z_trav)

            state = 5
        else:
            print("Invalid Dexarm Coordinates - please try again")
            state = 4

    # Returns to original search zone
    elif state == 5:

        print("Inside state 5")
        dexarm.soft_gripper_nature()
        dexarm.go_home()
        dexarm.fast_move_to(0, 300, z_search)
        object_found = False
        time.sleep(10)

        state = 1


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
