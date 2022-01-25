import cv2 as cv

cap = cv.VideoCapture(0)
eye_cascade = cv.CascadeClassifier('haar_eye.xml')


while True :
    isTrue , frame = cap.read()

    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    eye_roi = eye_cascade.detectMultiScale(gray , 1.1 , 5)

    for (x,y,w,h) in eye_roi:
        cv.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , thickness=2)
        cv.putText(frame , "Eye Detected" , (x,y-5) , cv.FONT_HERSHEY_COMPLEX , 1.0 , (255,255,255) , thickness=1)

    cv.imshow("Eye Detection",frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
