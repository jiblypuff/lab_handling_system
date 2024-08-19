# import the opencv library
import cv2
# import the 
from pydexarm import Dexarm
# import math
import math
import time

import yoloLib

z_search = 110

# define a dexarm object to control the Dexarm
dexarm = Dexarm("/dev/tty.usbmodem2078399E47531")
dexarm.go_home()
curr_pos = dexarm.get_current_position()
print('HOME: ', curr_pos)
dexarm.move_to(curr_pos[0], curr_pos[1], z_search)
dexarm.soft_gripper_nature()

# define a video capture object
vid = cv2.VideoCapture(0)

state = 1
# represents the center of the object to be picked and placed
center_X = 0
center_Y = 0
# represents where the object needs to be located in the frame to be picked properly
goal_x = 1001
goal_y = 310
# represents the height where the air picker will be just above the table surface
z_table = -110
# represents the height at which the arm will traverse across the table
z_trav = 50

# search rad
search_rad = 300

error_bound = 25
object_found = False
count = 0

x_lower_lim = -395
x_upper_lim = 395
y_lower_lim = 0
y_upper_lim = 395

slope_x = -59/305
slope_y = 41/194
sleep_ratio = 0.00897247


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

def movement_finished(x_coord, y_coord, x_target, y_target):
    print('cur pos: (', x_coord, ', ', y_coord, ')')
    print('target pos: (', x_target, ', ', y_target, ')')
    return abs(x_coord - x_target < 2) and abs(y_coord - y_target < 2)

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
    
prev_X = -1
prev_Y = -1
prev_dist = -1
wait_count = 0

check_input = input("Do you want to detect transparent objects? (y/n): ").lower()
while not (check_input == 'y' or check_input == 'n'):
    print("please input the letters y or n")
    check_input = input("Do you want to detect transparent objects? (y/n): ").lower()

check_transparent = check_input == 'y'


while(True):
    print("state="+str(state))

    # tup = yolo.getClosestObject(vid)
    # print(tup[0], tup[1])


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
    elif state == 1:
        # get center and dist of object closest to center
        center_X, center_Y, dist = yoloLib.getClosestObject(vid, check_transparent)
        print('DIST: ', dist)
        
        if (center_X != -1):
           object_found = True
           state = 2
        else:
            rotate_angle(-1)
            time.sleep(0.25)
            object_found = False
        
        
    elif state == 2:
        count = 0
        # if (check_arm_pos(center_X, center_Y)):
        #     state = 3
        #     print("Arm in correct position")
        # else:
        #     x_error = goal_x - center_X
        #     y_error = goal_y - center_Y
        #     print("Y error: " + str(y_error))

        #     if (abs(x_error) > error_bound):
        #         rotate_angle(x_error)
        #         if (abs(x_error) < 2 * error_bound):
        #             time.sleep(0.4)
        #         else:
        #             time.sleep(0.4)
        #     if (abs(y_error) > error_bound):
        #         move_radially(y_error)
        #         if (abs(y_error) < 2 * error_bound):
        #             time.sleep(0.5)
        #         else:
        #             time.sleep(0.5)
            
        #     state = 1

        # print('diff', goal_x - center_X, goal_y - center_Y)
        
        delta_arm_x = (goal_x - center_X) * slope_x
        delta_arm_y = (goal_y - center_Y) * slope_y
        curr_pos = dexarm.get_current_position()
        target_arm_x = curr_pos[0] + delta_arm_x
        target_arm_y = curr_pos[1] + delta_arm_y
        print(target_arm_x, target_arm_y)
        if not valid_location(target_arm_x, target_arm_y):
            print('Closest object is out of arm range')
            state = 1

        dexarm.fast_move_to(target_arm_x, target_arm_y, z_search)
        while(not movement_finished(curr_pos[0], curr_pos[1], target_arm_x, target_arm_y)): 
            curr_pos = dexarm.get_current_position()
        time.sleep(2)

        center_X, center_Y, dist = yoloLib.getClosestObject(vid, check_transparent)
        # time.sleep(0.5)
        x_error = goal_x - center_X
        y_error = goal_y - center_Y
        if (check_arm_pos(center_X, center_Y)):
            state = 3
        else:
            state = 1

    
    # Picks up the object and moves it to the drop zone
    elif state == 3:
        curr_pos = dexarm.get_current_position()
        print(curr_pos)
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
            dexarm.fast_move_to(drop_x, drop_y, z_search)
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
