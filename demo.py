import cv2
import numpy as np
from Traffic_Light_Classifier import pred_red_green_yellow


class TrafficLight():
    def __init__(self, position, zoom_ratio=1.0):
        # in position: 4-d vector, [xL,yT,xR,yB], define the position of the traffic light
        # zoom-ratio: default is 1. Sometimes the image is resized for visualization or other process,
        #             in that case, the predifined position need to fit the resized image.

        self.position = (np.asarray(position) / zoom_ratio).astype(int)
        self.process_size = (20,50)
        self.isDebug = False
        return

    def showPosition(self, frame, color=(255, 255, 255)):
        frame = cv2.rectangle(frame, tuple(self.position[:2]), tuple(self.position[2:]), color, 2)
        return frame

    def cropfromimage(self, img):
        x = self.position[0]
        y = self.position[1]
        w = self.position[2] - self.position[0]
        h = self.position[3] - self.position[1]
        crop_img = img[y:y + h, x:x + w]
        return crop_img

    def getSignal(self, frame, isDebug=False):
        # return: signal_txt, probability_float
        # five classes: unknown, red, yellow, green, red==>yellow (for hong kong)

        light_region = self.cropfromimage(frame)
        if self.isDebug: cv2.imshow('traffic light', light_region)
        traffic_path_std = cv2.resize(light_region, self.process_size)
        signal, prob = pred_red_green_yellow(traffic_path_std, isDebug)  # red yellow green
        return signal, prob

    def showResult(self, frame, isLabelBottom=False, isDebug=False, color=(0, 255, 0)):
        signal, prob = self.getSignal(frame, isDebug)
        s = signal + ':%.2f' % (prob)
        if isLabelBottom:  # sometimes the traffic lights are crowd, the text need to label at different position to avoid overlaping
            frame = cv2.putText(frame, s, (self.position[0], self.position[3] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color, 2)
        else:
            frame = cv2.putText(frame, s, tuple(self.position[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        frame = self.showPosition(frame, color)
        return frame


if __name__ == '__main__':
    img_in = cv2.imread("./test_img.jpg")

    # define the position of the traffic light: [xL,yT,xR,yB]
    m_light_1 = TrafficLight(position=[20, 108, 103, 351])
    m_light_2 = TrafficLight(position=[166, 106, 255, 348])
    m_light_3 = TrafficLight(position=[313, 111, 412, 354])
    m_light_4 = TrafficLight(position=[472, 109, 568, 355])

    # visualize the result. To get the raw signal information, use the "getSignal" function.
    res_visual = m_light_1.showResult(img_in)
    res_visual = m_light_2.showResult(res_visual, isLabelBottom=True)
    res_visual = m_light_3.showResult(res_visual)
    res_visual = m_light_4.showResult(res_visual)

    # visualization and saving
    cv2.imshow('result', res_visual)
    cv2.imwrite('result.jpg', res_visual)
    cv2.waitKey(0)
