import numpy as np
import mss
import cv2
from functools import partial

from utils.particle import Particle, ParticlesGrid
from utils.orb_tracker import ORB_tracker


SMALL_TOP = 160
SMALL_LEFT = 2019
SMALL_WIDTH = 1280
SMALL_HEIGHT = 720
LARGE_TOP = 80
LARGE_LEFT = 1921
LARGE_WIDTH = 1901
LARGE_HEIGHT = 1135
youtube_tlwh_small = (160, 2019, 1280, 720)
youtube_tlwh_large = (80, 1921, 1901, 1135)

mouse_coords = (-1, -1)
kernel_size = 41
crop_size = kernel_size * 2 + 1
nn_size = 7
particles = [Particle(kernel_size, crop_size, nn_size, p=3, q=1e-9, temperature=0.1)]
particle_grid = ParticlesGrid(youtube_tlwh_small[2:], kernel_size, crop_size, nn_size, p=3, q=1e-9, temperature=0.1, grid_size=(2*8, 2*6))
orb_tracker = ORB_tracker(n_features=100, top_k_features=10, init_crop_size=101)


def track_mouse_clicked_target(tlwh=None, monitor_number=0):
    global mouse_coords, particles

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, particles
        if event == cv2.EVENT_LBUTTONUP:
            mouse_coords = (x, y)
            # convert img to grayscale
            particles[0].reset()
            particles[0].create_kernel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mouse_coords)

    if tlwh is None:
        tlwh = (SMALL_TOP, SMALL_LEFT, SMALL_WIDTH, SMALL_HEIGHT)
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        # The screen part to capture
        monitor = {
            "top": mon["top"] + tlwh[0],  # 100px from the top
            "left": mon["left"] + tlwh[1],  # 100px from the left
            "width": tlwh[2],
            "height": tlwh[3],
            "mon": monitor_number,
        }
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', mouse_callback)
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if particles[0].kernel is not None:
                # cv2.imshow("kernel", (255 * kernel/kernel.max()).astype(np.uint8))
                mouse_coords, max_chance = particles[0].update(img_gray)
                if max_chance < 1e-3:
                    particles[0].reset()
                    print("reset", max_chance*100, "%")
                    mouse_coords = (-1, -1)
            img_show = img.copy()
            if mouse_coords[0] > -1:
                # cv2.arrowedLine(img_show, np.array(mouse_coords)-velocity_vector, np.array(mouse_coords), (0, 255, 0), 2)
                cv2.circle(img_show, mouse_coords, kernel_size//2, (0, 255, 0), 2)
            cv2.imshow('frame', img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def track_mouse_clicked_target_ORB(tlwh=None, monitor_number=0):
    global mouse_coords

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords
        if event == cv2.EVENT_LBUTTONUP:
            mouse_coords = (x, y)
            orb_tracker.reset(img_gray, mouse_coords)

    if tlwh is None:
        tlwh = (SMALL_TOP, SMALL_LEFT, SMALL_WIDTH, SMALL_HEIGHT)

    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        # The screen part to capture
        monitor = {
            "top": mon["top"] + tlwh[0],  # 100px from the top
            "left": mon["left"] + tlwh[1],  # 100px from the left
            "width": tlwh[2],
            "height": tlwh[3],
            "mon": monitor_number,
        }
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', mouse_callback)
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if orb_tracker.xy is not None:
                pixels = orb_tracker.update(img_gray)
                # cv2.arrowedLine(img_show, np.array(mouse_coords)-velocity_vector, np.array(mouse_coords), (0, 255, 0), 2)
                # draw rectangle around the target as a bounding box with the size of the crop
                cv2.rectangle(img, (orb_tracker.xy[0]-orb_tracker.next_crop_size[0]//2, orb_tracker.xy[1]-orb_tracker.next_crop_size[1]//2),
                                (orb_tracker.xy[0]+orb_tracker.next_crop_size[0]//2, orb_tracker.xy[1]+orb_tracker.next_crop_size[1]//2), (0, 255, 0), 2)
                for pixel in pixels:
                    cv2.circle(img, tuple(pixel[0]), 2, (0, 255, 0), 2)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def track_grid(tlwh=None, monitor_number=0):
    global particle_grid
    if tlwh is None:
        tlwh = (SMALL_TOP, SMALL_LEFT, SMALL_WIDTH, SMALL_HEIGHT)
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        # The screen part to capture
        monitor = {
            "top": mon["top"] + tlwh[0],  # 100px from the top
            "left": mon["left"] + tlwh[1],  # 100px from the left
            "width": tlwh[2],
            "height": tlwh[3],
            "mon": monitor_number,
        }
        cv2.namedWindow('frame')
        img_byte = sct.grab(monitor)
        img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        particle_grid.create_kernels(img_gray)
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            particle_grid.update(img_gray)
            particle_grid.drawParticles(img)
            particle_grid.drawVelocities(img)
            particle_grid.drawKernelsWindows(img)
            cv2.imshow('frame', img)
            particle_grid.reset()
            particle_grid.create_kernels(img_gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def frame_to_numpy(frame, height, width):
    img = np.frombuffer(frame.rgb, np.uint8).reshape(height, width, 3)[:, :, ::-1]
    return img.astype(np.uint8)

if __name__ == '__main__':
    # track_mouse_clicked_target(youtube_tlwh_large)
    # track_grid(youtube_tlwh_small)
    track_mouse_clicked_target_ORB(youtube_tlwh_large)