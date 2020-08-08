import numpy as np
import cv2


# The function is modified based on the package "https://github.com/thedch/traffic-light-classifier"
# The modification contains: we refine the decision process, of which:
#       1) the prediction is a soft decision that is based on probability.
#       2) the thresholds of Hue, Saturation, Brightness are refined for Hong Kong trasportation system.
#          You may define your threshold based on your cases.
#       3) refined definition of hue (red has two parts).

def create_feature(rgb_image):
    '''Basic brightness feature, required by Udacity'''
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space

    sum_brightness = np.sum(hsv[:, :, 2])  # Sum the brightness values
    area = 32 * 32
    avg_brightness = sum_brightness / area  # Find the average

    return avg_brightness


def high_saturation_pixels(rgb_image, threshold):
    '''Returns average red and green content from high saturation pixels'''
    high_sat_pixels = []
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    for i in range(32):
        for j in range(32):
            if hsv[i][j][1] > threshold:
                high_sat_pixels.append(rgb_image[i][j])

    if not high_sat_pixels:
        return highest_sat_pixel(rgb_image)

    sum_red = 0
    sum_green = 0
    for pixel in high_sat_pixels:
        sum_red += pixel[0]
        sum_green += pixel[1]

    # TODO: Use sum() instead of manually adding them up
    avg_red = sum_red / len(high_sat_pixels)
    avg_green = sum_green / len(high_sat_pixels) * 0.8  # 0.8 to favor red's chances
    return avg_red, avg_green


def highest_sat_pixel(rgb_image):
    '''Finds the higest saturation pixel, and checks if it has a higher green
    content, or a higher red content'''

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]

    x, y = (np.unravel_index(np.argmax(s), s.shape))
    if rgb_image[x, y, 0] > rgb_image[x, y, 1] * 0.9:  # 0.9 to favor red's chances
        return 1, 0  # Red has a higher content
    return 0, 1


def findNonZero(rgb_image):
    rows, cols, _ = rgb_image.shape
    counter = 0

    for row in range(rows):
        for col in range(cols):
            pixel = rgb_image[row, col]
            if sum(pixel) != 0:
                counter = counter + 1

    return counter


def pred_red_green_yellow(bgr_image, isdebug=False, process_size=(20, 50)):
    '''Determines the Red, Green, and Yellow content in each image using HSV and
    experimentally determined thresholds. Returns a classification based on the
    values.
    The hue value is ranged from 0 to 180 defined by OpenCV, where it is a half of the real hue value.
    '''
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:, :, 1])  #
    sum_bright = np.sum(hsv[:, :, 2])

    area = float(process_size[0] * process_size[1])
    # avg_saturation = sum_saturation / area  # Find the average
    avg_bright = sum_bright / area

    sat_low = 25  # int(avg_saturation * 1.3)
    val_low = int(avg_bright * 1.3)  # 140

    # green (may need further refinement for your case)
    lower_green = np.array([70, sat_low, val_low])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask=green_mask)

    # Yellow (may need further refinement for your case)
    lower_yellow = np.array([5, sat_low, val_low])
    upper_yellow = np.array([20, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask)

    # Red (has two Hue range, may need further refinement for your case)
    lower_red = np.array([160, sat_low, val_low])
    upper_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result_1 = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)

    lower_red = np.array([0, sat_low, val_low])
    upper_red = np.array([5, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result_2 = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)

    sum_green = findNonZero(green_result)
    sum_yellow = findNonZero(yellow_result)
    sum_red = findNonZero(red_result_1) + findNonZero(red_result_2)

    sum = float(sum_green + sum_yellow + sum_red)
    if sum < 10: return ['Unknow', 0]  # low-confidence prediction

    score = np.asarray([sum_red, sum_yellow, sum_green])
    prob = score / np.sum(score)
    if isdebug: print(score, prob)
    decision = np.argmax(prob)

    out_prob = prob[decision]
    label = ['Red', "Yellow", "Green"]

    # soft decision
    if out_prob > 0.8:
        return [label[int(decision)], prob[decision]]
    elif (prob[0] + prob[1]) > 0.8:
        return ['Red=>Yellow', prob[0] + prob[1]]
    else:
        return [label[int(decision)], prob[decision]]
