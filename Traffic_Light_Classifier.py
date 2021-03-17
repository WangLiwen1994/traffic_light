import numpy as np
import cv2


# The function is modified based on the package "https://github.com/thedch/traffic-light-classifier"
# The modification contains: we refine the decision process, of which:
#       1) the prediction is a soft decision that is based on probability.
#       2) the thresholds of Hue, Saturation, Brightness are refined for Hong Kong trasportation system.
#          You may define your threshold based on your cases.
#       3) refined definition of hue (red has two parts).
# 2021-03-16, liwen modified the threshold and speed up calculation

class Traffic_light_classifier():
    def __init__(self, nSignal=3, process_size=(20, 50), isVideoBased=True):
        self.Pix_A_Light = (50 - 10) * (20 - 5) / 3 * 1.3
        self.Min_Pix_A_Light = (50 - 10) * (20 - 5) / 3 * 0.8
        self.kernel = np.ones((7, 7), np.uint8)
        self.nSignal = nSignal
        self.process_size = process_size
        self.area = process_size[0] * process_size[1]
        self.previous_signal = None
        self.avg_bright = None
        self.isVideoBased = isVideoBased

    def findNonZero(self, rgb_image):
        if rgb_image.ndim == 3:
            rgb_sum = np.sum(rgb_image, axis=2)
            return np.count_nonzero(rgb_sum)
        else:
            return np.count_nonzero(rgb_image)

    def check_Yellow_Region(self, yellow_mask):
        if self.nSignal != 3: return False
        real_yellow_mask = yellow_mask[int(self.process_size[1] * 2 / 5):int(self.process_size[1] * 3 / 5) + 1, :]
        if self.findNonZero(real_yellow_mask) > self.process_size[0] // 2:
            return True
        else:
            return False

    def pred_red_green_yellow(self, bgr_image, isdebug=False):
        '''Determines the Red, Green, and Yellow content in each image using HSV and
        experimentally determined thresholds. Returns a classification based on the
        values.
        The hue value is ranged from 0 to 180 defined by OpenCV, where it is a half of the real hue value.
        '''
        # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        # avg_saturation = sum_saturation / area  # Find the average
        avg_bright = np.mean(hsv[:, :, 2]).mean()
        if self.avg_bright is None: self.avg_bright = avg_bright
        self.avg_bright = (0.01 * avg_bright + 0.99 * self.avg_bright)

        sat_low = 25  # int(avg_saturation * 1.3)
        val_low = int(self.avg_bright * 1.5)  # 140

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

        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)

        sum_green = self.findNonZero(green_mask)
        sum_yellow = self.findNonZero(yellow_mask)
        sum_red = self.findNonZero(red_mask)

        sum = float(sum_green + sum_yellow + sum_red)
        if sum < 10: return ['Unknow', 0]  # low-confidence prediction

        if self.nSignal == 2:
            red_mask += yellow_mask
            yellow_mask[...] = 0
            sum_red += sum_yellow
            sum_yellow = 0

        score = np.asarray([sum_red, sum_yellow, sum_green])
        prob = score / np.sum(score)
        if isdebug: print(score, prob)
        decision = np.argmax(prob)

        out_prob = prob[decision]
        label = ['Red', "Yellow", "Green"]

        # soft decision
        if out_prob > 0.9:  # high confidence
            self.previous_signal = label[int(decision)]
            return [label[int(decision)], prob[decision]]
        elif self.isVideoBased and prob[1] > 0.15 and (prob[0] + prob[1]) > 0.7 and self.check_Yellow_Region(
                yellow_mask) and (self.previous_signal == "Red" or self.previous_signal == "Red=>Yellow"):
            # cv2.waitKey(500)
            self.previous_signal = 'Red=>Yellow'
            return ['Red=>Yellow', prob[0] + prob[1]]
        elif (not self.isVideoBased) and prob[1] > 0.15 and (prob[0] + prob[1]) > 0.7 and self.check_Yellow_Region(
                yellow_mask):
            self.previous_signal = 'Red=>Yellow'
            return ['Red=>Yellow', prob[0] + prob[1]]
        elif label[int(decision)] == "Yellow" and self.check_Yellow_Region(yellow_mask):
            self.previous_signal = "Yellow"
            return ["Yellow", out_prob]
        else:  # low confidence
            if prob[0] + prob[1] > prob[2]:
                self.previous_signal = "Red"
                return ['Red', prob[0] + prob[1]]
            else:
                self.previous_signal = "Green"
                return ['Green', prob[2]]
