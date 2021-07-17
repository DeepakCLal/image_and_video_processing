import cv2
import matplotlib.pyplot as plt
import numpy as np

vid = cv2.VideoCapture('train.mp4')

while True:
  ret, frame = vid.read()
  if ret == True:
    cv2.imshow('Original', frame)
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img,mask)
        return masked_image
    def draw_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=3)
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img
    temp_image = cv2.imwrite('test.jpg',frame)
    image = cv2.imread('test.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (-100 , height),
        (1200, height),
        (290,220)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.GaussianBlur(gray_image, (3,3), 0)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32), )
    cv2.imshow('Gray', cropped_image)
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=5, lines=np.array([]), minLineLength=5,
                            maxLineGap=20)
    print(lines)
    image_with_lines = draw_the_lines(frame, lines)
    cv2.imshow('final', image_with_lines)
    if cv2.waitKey(25) & 0XFF == ord('q'):
      break
  else:
    break
