import numpy as np
import cv2


# The function is modified based on the package "https://github.com/thedch/traffic-light-classifier"
# The modification contains: we refine the decision process, of which:
#       1) the prediction is a soft decision that is based on probability.
#       2) the thresholds of Hue, Saturation, Brightness are refined for Hong Kong trasportation system.
#          You may define your threshold based on your cases.
#       3) refined definition of hue (red has two parts).
# 2021-03-16, liwen modified the threshold and speed up calculation

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
    # 2021-03-13, Liwen modified for fast speed
    if rgb_image.ndim == 3:
        rgb_sum = np.sum(rgb_image, axis=2)
        return np.count_nonzero(rgb_sum)
    else:
        return np.count_nonzero(rgb_image)


Pix_A_Light = (50-10)*(20-5)/3*1.3
Min_Pix_A_Light = (50-10)*(20-5)/3*0.8

def pred_red_green_yellow(bgr_image, isdebug=False, process_size=(20, 50)):
    global Pix_A_Light
    '''Determines the Red, Green, and Yellow content in each image using HSV and
    experimentally determined thresholds. Returns a classification based on the
    values.
    The hue value is ranged from 0 to 180 defined by OpenCV, where it is a half of the real hue value.
    '''
    # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    sum_bright = np.sum(hsv[:, :, 2])

    area = float(process_size[0] * process_size[1])
    # avg_saturation = sum_saturation / area  # Find the average
    avg_bright = sum_bright / area

    sat_low = 25  # int(avg_saturation * 1.3)
    val_low = int(avg_bright * 1.3)  # 140
    kernel = np.ones((7, 7), np.uint8)

    # green (may need further refinement for your case)
    lower_green = np.array([60, sat_low, val_low])
    upper_green = np.array([90, 256, 256])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Yellow (may need further refinement for your case)
    lower_yellow = np.array([5, sat_low, val_low])
    upper_yellow = np.array([30, 256, 256])
    yellow_mask_1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = yellow_mask_1

    # Red (has two Hue range, may need further refinement for your case)
    lower_red = np.array([160, sat_low, val_low])
    upper_red = np.array([180, 256, 256])
    red_mask_1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([0, sat_low, val_low])
    upper_red = np.array([7, 256, 256])
    red_mask_2 = cv2.inRange(hsv, lower_red, upper_red)
    red_mask = red_mask_1 + red_mask_2

    green_mask= cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    sum_green = findNonZero(green_mask)
    sum_yellow = findNonZero(yellow_mask)
    sum_red = findNonZero(red_mask)

    sum = float(sum_green + sum_yellow + sum_red)
    if sum < 10: return ['Unknow', 0]  # low-confidence prediction

    score = np.asarray([sum_red, sum_yellow, sum_green])
    prob = score / np.sum(score)
    if isdebug: print(score, prob)
    decision = np.argmax(prob)

    out_prob = prob[decision]
    label = ['Red', "Yellow", "Green"]

    # soft decision
    if out_prob > 0.95:  # high confidence
        if (label[int(decision)] == 'Red'):
            Pix_A_Light = max((score[int(decision)]+50*Pix_A_Light)/51, Min_Pix_A_Light)
        return [label[int(decision)], prob[decision]]
    elif Pix_A_Light is not None and (score[0] + score[1]) > 1.1*Pix_A_Light and prob[1] > 0.1 and (
            prob[0] + prob[1]) > 0.8:
        return ['Red=>Yellow', prob[0] + prob[1]]
    elif label[int(decision)] == "Yellow" and out_prob > 0.55:
        return ["Yellow", out_prob]
    else:
        return [label[int(decision)], prob[decision]]
