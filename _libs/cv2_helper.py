#TODO: modificare tutti i rif di cv2 in questo modo ( dovrebbe essere più veloce)
from cv2 import namedWindow as cv2_namedWindow
from cv2 import moveWindow as cv2_moveWindow
from cv2 import resize as cv2_resize
from cv2 import vconcat as cv2_vconcat
from cv2 import hconcat as cv2_hconcat


def resizeInWidth(frame,width):
    original_height, original_width = frame.shape[:2]
    # Set the desired width of the resized frame

    # Calculate the aspect ratio of the original frame
    aspect_ratio = original_height / original_width

    # Calculate the height of the resized frame
    desired_height = int(width * aspect_ratio)

    # Resize the frame
    return cv2_resize(frame, (width, desired_height))
def resizeInHeigth(frame,height):
    original_height, original_width = frame.shape[:2]
    # Set the desired width of the resized frame

    # Calculate the aspect ratio of the original frame
    aspect_ratio = original_height / original_width

    # Calculate the height of the resized frame
    desired_width = int(height / aspect_ratio)

    # Resize the frame
    return cv2_resize(frame, (height, desired_width))
def resize(frame,height,width):
    # Resize the frame
    return cv2_resize(frame, (height, width))



def concat_tile(im_list_2d):
    return cv2_vconcat([cv2_hconcat(im_list_h) for im_list_h in im_list_2d])
