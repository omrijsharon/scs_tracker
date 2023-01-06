import cv2
from multiprocessing import Process, Queue

from core.pipelines import capture_frames_live, load_frame

youtube_tlwh_small = (160, 2019, 1280, 720)
youtube_tlwh_large = (80, 1921, 1901, 1135)


class ProcessManager:
    def __init__(self):
        self.processes = []

    def start(self, target, args):
        q = Queue()
        p = Process(target=target, args=(q, ) + args)
        p.start()
        self.processes.append((p, q))

    def join(self):
        for p, q in self.processes:
            p.join()


if __name__ == '__main__':
    tlwh = youtube_tlwh_small
    q = Queue()
    p = Process(target=capture_frames_live, args=(q, 0, tlwh))
    p.start()
    while True:
        img = load_frame(q, tlwh[3], tlwh[2])
        # opencv show image:
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()