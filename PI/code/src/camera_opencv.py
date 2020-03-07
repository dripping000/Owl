from multiprocessing import Queue
import requests

import cv2

from base_camera import BaseCamera


SERVER_IP = "http://192.168.0.113:90"


# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
def Queue_frame(inputQueue, outputQueue):
    Exception_num = 0
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            print("Queue_frame")
            start = cv2.getTickCount()

            data = {}
            pos = [0, 0, 0, 0]
            try:
                frame = inputQueue.get()
                file_data = {'image': cv2.imencode('.jpg', frame)[1].tobytes()}

                # 获取人脸坐标
                r = requests.post(SERVER_IP+"/face_detect", files=file_data)
                pos = eval(r.text)

            except Exception as result:
                print(result)
                Exception_num = Exception_num + 1

            end = cv2.getTickCount()
            time = (end - start) / cv2.getTickFrequency()
            print("time:", time)
            print("Exception_num:", Exception_num)

            data['pos'] = pos
            print(data)
            outputQueue.put(data)


class CameraOpenCV(BaseCamera):

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        face_x1 = 0
        face_y1 = 0
        face_x2 = 0
        face_y2 = 0

        while True:
            # read current frame
            _, src = camera.read()
            src_tmp = src.copy()

            if inputQueue.empty():
                print("inputQueue.empty()")
                inputQueue.put(src_tmp)

            if not outputQueue.empty():
                print("not outputQueue.empty()")
                data = outputQueue.get()
                pos = data['pos']

                height = src.shape[0]
                width = src.shape[1]

                face_x1 = int(pos[0] * width)
                face_y1 = int(pos[1] * height)
                face_x2 = int(pos[2] * width)
                face_y2 = int(pos[3] * height)

            # 绘制人脸框
            cv2.rectangle(src, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)

            cv2.waitKey(100)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', src)[1].tobytes()
