import os
import numpy as np
import cv2
from scipy.io import savemat

INPUT_FOLDER = 'DJI_0118_to_be_annotated_only'
OUTPUT_FOLDER = 'DJI_0118_to_be_annotated_only_mat'

CLASS_COLORS_RGB =((0,255,0),(0,127,0),(255,255,0),(255,127,0),(255,255,255),(255,0,255),(127,127,127),(0,0,255),(0,255,255),(127,127,63),(255,0,0),(127,127,0))
CLASS_NAMES = ('land', 'forest', 'residential', 'capitze', 'road', 'church', 'cars', 'water', 'sky', 'hill', 'person', 'fence')

NUM_CLASSES = len(CLASS_NAMES)


input_files = os.listdir(INPUT_FOLDER)



for input_file in input_files:
    if not 'seg_' in input_file or not '.png' in input_file:
        continue
    print(input_file)

    current_image_rgb = cv2.imread(os.path.join(INPUT_FOLDER, input_file))
    mat_dict = {}
    mat_dict['classes'] = []
    mat_dict['polygons'] = []
    for idx_class in range(NUM_CLASSES):
        bw_poly = np.zeros_like(np.squeeze(current_image_rgb[:,:,0]))
        #print(bw_poly.shape)

        #current_image_rgb = cv2.cvtColor(current_image_rgb, cv2.COLOR_BGR2RGB)

        bw_poly = cv2.inRange(current_image_rgb, CLASS_COLORS_RGB[idx_class][::-1], CLASS_COLORS_RGB[idx_class][::-1])

        cv2.imwrite("class_"+str(idx_class)+".png", bw_poly)
        #print(np.count_nonzero(bw_poly))
        bin, contours, hierarchy = cv2.findContours(np.array(bw_poly, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            approxContourPoints = cv2.approxPolyDP(np.asarray(cnt), 2, closed=True)
            if len(approxContourPoints) <= 2:
                continue
            mat_dict['classes'].append(float(idx_class+1))
            approxContourPointsRightShape = np.zeros((len(approxContourPoints), 2))
            for idx, point in enumerate(approxContourPoints):
                approxContourPointsRightShape[idx, 0] = point[0, 0]+1
                approxContourPointsRightShape[idx, 1] = point[0, 1]+1
            mat_dict['polygons'].append(approxContourPointsRightShape)
            #print(approxContourPoints.shape)
            #print(len(approxContourPoints))
            #print(approxContourPointsRightShape.shape)
    #print(mat_dict)
    mat_dict['polygons'] = np.array(mat_dict['polygons'])
    savemat(os.path.join(OUTPUT_FOLDER, input_file.replace('.png', '.mat')), mat_dict)
