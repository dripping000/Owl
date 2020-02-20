import cv2

from base_camera import BaseCamera


class CameraOpenCV(BaseCamera):

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, src = camera.read()

            cv2.waitKey(100)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', src)[1].tobytes()
