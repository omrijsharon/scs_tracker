import numpy as np
import mss
import cv2
from functools import partial

from utils.particle import Particle, ParticlesGrid
from utils.orb_tracker import ORB_tracker

from utils.helper_functions import frame_to_numpy

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
n_particles = 5
smallest_kernel = 31
kernel_size = np.arange(smallest_kernel//10, smallest_kernel//10 + n_particles) * 10 + 1
crop_size = (kernel_size.max() * 2).astype(int)
crop_size = (crop_size//2) + 1
nn_size = 3
particles = []
# particle_grid = ParticlesGrid(youtube_tlwh_small[2:], kernel_size, crop_size, nn_size, p=3, q=1e-9, temperature=0.1, grid_size=(2*8, 2*6))
# orb_tracker = ORB_tracker(n_features=1000, top_k_features=100, init_crop_size=201)


def track_mouse_clicked_target(tlwh=None, monitor_number=0):
    global mouse_coords, particles, n_particles

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, particles, n_particles
        if event == cv2.EVENT_LBUTTONUP:
            mouse_coords = (x, y)
            # convert img to grayscale
            for krnl_sz in kernel_size:
                xy = tuple(np.array(mouse_coords) + 0 * np.random.randint(-crop_size//4, crop_size//4, 2))
                particles.extend([Particle(krnl_sz, crop_size, nn_size, p=3, q=1e-9, temperature=0.1)])
                particles[-1].reset()
                particles[-1].create_kernel((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2, xy)
                particles[-1].rembg = True

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
        xy_array = np.zeros((n_particles, 2))
        particles2delete = []
        particles2duplicate = []
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2
            if len(particles) > 0:
                # cv2.imshow("kernel", (255 * kernel/kernel.max()).astype(np.uint8))
                if particles[0].coordinates is not None:
                    for i, particle in enumerate(particles):
                        # if particle coordinates are out of image, delete particle
                        if particle.coordinates[0] < img_gray.shape[1]*0.1 or particle.coordinates[0] >= img_gray.shape[1]*0.9 or particle.coordinates[1] < img_gray.shape[0] * 0.1 or particle.coordinates[1] >= 0.9 * img_gray.shape[0]:
                            particles2delete.append(i)
                            continue
                        # xy, max_chance = particle.update(img_gray)
                        xy, max_chance = particle.update(img_gray)
                        if xy is None:
                            particles2delete.append(i)
                            continue
                        xy_array[i] = xy
                        if max_chance * 100 > 0.2:
                        # if np.sqrt(np.prod(np.array(std))) < 50:
                        #     particles2duplicate.append([i])
                            cv2.putText(img, "{:.1f}".format(100*max_chance), (xy[0], xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            # draw rectangle around xy sized kernel_size with color proportional to max_chance
                            cv2.rectangle(img, (xy[0] - particle.kernel_size//2, xy[1] - particle.kernel_size//2), (xy[0] + particle.kernel_size//2, xy[1] + particle.kernel_size//2), (0, 255 * max_chance, 255 * max_chance), 2)
                            # cv2.circle(img, xy, particle.kernel_size // 2, (0, 255, 0), 2)
                            # draw rectangle with center at xy+particle.velocity and size particle.crop_size:
                            cv2.rectangle(img, (int(xy[0]+particle.velocity[0] - particle.crop_size // 2), int(xy[1]+particle.velocity[1] - particle.crop_size // 2)), (int(xy[0]+particle.velocity[0] + particle.crop_size // 2), int(xy[1]+particle.velocity[1] + particle.crop_size // 2)), (0, 127, 0), 2)
                        # put test of max chance *100 with only 2 digits after the dot
                        # cv2.putText(img, "{:.2f}".format(np.sqrt(np.prod(np.array(std)))), (xy[0], xy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # cv2.putText(img, "{:.2f}".format(max_chance * 100), (xy[0], xy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            particles2delete.append(i)
                    for i in particles2delete[::-1]:
                        particles.pop(i)
                    particles2delete = []
                    # for i in particles2duplicate:
                    #     particle = Particle(kernel_size, crop_size, nn_size, p=3, q=1e-9, temperature=0.1)
                    #     xy = tuple(np.array(particles[i].coordinates) + np.random.randint(-crop_size // 2, crop_size // 2, 2))
                    #     particle.reset()
                    #     particle.create_kernel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), xy)
                    #     particles.append(particle)
                    # if max_chance < 1e-3:
                    #     particles[0].reset()
                    #     print("reset", max_chance*100, "%")
                    #     mouse_coords = (-1, -1)
            cv2.imshow('frame', img)
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
                # pixels = orb_tracker.update_cosim(img_gray)
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

if __name__ == '__main__':
    track_mouse_clicked_target(youtube_tlwh_small)
    # track_grid(youtube_tlwh_small)
    # track_mouse_clicked_target_ORB(youtube_tlwh_large)