import cv2

video = cv2.VideoCapture('cars1.mp4')

cars_cascade = cv2.CascadeClassifier('cars.xml')

while 1:
    ret, frames = video.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    cars = cars_cascade.detectMultiScale(gray, 1.05, 5)

    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]

    cv2.imshow('video',frames)

    if cv2.waitKey(33)==27:
        break

cv2.destroyAllWindows()

