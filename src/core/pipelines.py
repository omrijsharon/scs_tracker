import numpy as np
import mss
import cv2
from functools import partial

from utils.particle import Particle, ParticlesGrid, ParticleRembg
from utils.orb_tracker import ORB_tracker

from utils.helper_functions import frame_to_numpy, particles_mean_std, get_particles_coordinates

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
n_particles = 13
smallest_kernel = 31
# kernel_size = np.arange(smallest_kernel//10, smallest_kernel//10 + n_particles) * 10 + 1
kernel_size = np.ones(n_particles, dtype=np.int32) * smallest_kernel
crop_size = (kernel_size.max() * 3).astype(int)
crop_size = (crop_size//2) + 1
nn_size = 3
particles = []
# particle_grid = ParticlesGrid(youtube_tlwh_small[2:], kernel_size, crop_size, nn_size, p=3, q=1e-9, temperature=0.1, grid_size=(2*8, 2*6))
# orb_tracker = ORB_tracker(n_features=1000, top_k_features=100, init_crop_size=201)


def track_mouse_clicked_target_with_rembg(tlwh=None, monitor_number=0):
    global mouse_coords, particles, n_particles

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, particles, n_particles
        if event == cv2.EVENT_LBUTTONUP:
            print("Mouse clicked at: ", x, y)
            mouse_coords = (x, y)
            # convert img to grayscale
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # xy = tuple(np.array(mouse_coords))
            particles.extend([ParticleRembg()])
            particles[-1].reset(frame_gray)

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
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(particles) > 0:
                if particles[0].coordinates is not None:
                    for i, particle in enumerate(particles):
                        # if particle coordinates are out of image, delete particle
                        if particle.coordinates[0] < img_gray.shape[1]*0.1 or particle.coordinates[0] >= img_gray.shape[1]*0.9 or particle.coordinates[1] < img_gray.shape[0] * 0.1 or particle.coordinates[1] >= 0.9 * img_gray.shape[0]:
                            particles2delete.append(i)
                            continue
                        xy, certainty = particle.update(img_gray)
                        if xy is None:
                            particles2delete.append(i)
                            continue
                        xy_array[i] = xy
                        if certainty * 100 > 1.0:
                            cv2.putText(img, "{:.1f}".format(100*certainty), (xy[0], xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            # draw particle bbox params:
                            x, y, w, h = particle.bbox_params
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            # draw circle at particle center:
                            cv2.circle(img, (particle.coordinates[0], particle.coordinates[1]), 2, (255, 255, 255), 2)
                            # draw rectangle with center at xy+particle.velocity and size particle.crop_size:
                            # FIX:
                            # cv2.rectangle(img, (int(xy[0]+particle.velocity[0] - particle.crop_size[0] // 2), int(xy[1]+particle.velocity[1] - particle.crop_size[1] // 2)), (int(xy[0]+particle.velocity[0] + particle.crop_size // 2), int(xy[1]+particle.velocity[1] + particle.crop_size // 2)), (0, 127, 0), 2)
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


def track_mouse_clicked_target(tlwh=None, monitor_number=0):
    global mouse_coords, particles, n_particles

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, particles, n_particles
        if event == cv2.EVENT_LBUTTONUP:
            mouse_coords = (x, y)
            if len(particles) > 0:
                for i, particle in enumerate(particles):
                    particle.reset()
                    xy = tuple(np.array(mouse_coords) + np.random.randint(-crop_size//2, crop_size//2, 2))
                    particle.create_kernel((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2, xy)
            else:
                for krnl_sz in kernel_size:
                    xy = tuple(np.array(mouse_coords) + 1 * np.random.randint(-crop_size//4, crop_size//4, 2))
                    particles.extend([Particle(krnl_sz, crop_size, nn_size, p=3, q=1e-9, temperature=0.05, pixel_noise=1)])
                    particles[-1].reset()
                    particles[-1].create_kernel((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2, xy)

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
        rect_debug = True
        mask = np.ones((n_particles,), dtype=bool)
        max_chances = np.zeros((n_particles,))
        particle_max_chance_threshold = 0.001 # under this value particle will be replaced
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2
            if len(particles) > 0:
                # cv2.imshow("kernel", (255 * kernel/kernel.max()).astype(np.uint8))
                if particles[0].coordinates is not None:
                    for i, particle in enumerate(particles):
                        # if particle coordinates are out of image, delete particle
                        if particle.coordinates[0] < img_gray.shape[1]*0.05 or particle.coordinates[0] >= img_gray.shape[1]*0.95 or particle.coordinates[1] < img_gray.shape[0] * 0.1 or particle.coordinates[1] >= 0.9 * img_gray.shape[0]:
                            mask[i] = False
                            continue
                        # xy, max_chance = particle.update(img_gray)
                        xy, max_chance = particle.update(img_gray)
                        if xy is None:
                            mask[i] = False
                            continue
                        max_chances[i] = max_chance
                        if rect_debug:
                            cv2.putText(img, "{:.1f}".format(100*max_chance), (xy[0], xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            # draw rectangle around xy sized kernel_size with color proportional to max_chance
                            cv2.rectangle(img, (xy[0] - particle.kernel_size//2, xy[1] - particle.kernel_size//2), (xy[0] + particle.kernel_size//2, xy[1] + particle.kernel_size//2), (0, 255 * max_chance, 255 * max_chance), 2)
                            # draw rectangle with center at xy+particle.velocity and size particle.crop_size:
                            cv2.rectangle(img, (int(xy[0]+particle.velocity[0] - particle.crop_size // 2), int(xy[1]+particle.velocity[1] - particle.crop_size // 2)), (int(xy[0]+particle.velocity[0] + particle.crop_size // 2), int(xy[1]+particle.velocity[1] + particle.crop_size // 2)), (0, 127, 0), 2)
                    mask[np.argwhere(max_chances < max_chances * particle_max_chance_threshold)] = False
                    particles_coordinates = get_particles_coordinates(particles)
                    coordinates_mean, coordinates_std = particles_mean_std(particles_coordinates)
                    # for each particle, check if its distance from the mean is more than 2 times the std:
                    # print(np.sqrt(np.sum(coordinates_std ** 2)))
                    particles_distances_from_mean = np.sqrt(np.sum((particles_coordinates - coordinates_mean) ** 2, axis=1))
                    mask[particles_distances_from_mean > min(1.8 * np.sqrt(np.sum(coordinates_std ** 2)), 2 * crop_size)] = False
                    coordinates_mean, coordinates_std = particles_mean_std(particles_coordinates, mask=mask)
                    num_particles_to_delete = sum(1-mask)
                    if num_particles_to_delete > 0 and num_particles_to_delete < n_particles:
                        for i in np.argwhere(mask==False).flatten():
                            xy = tuple((coordinates_mean + 1.0 * np.random.randint(-crop_size // 2, crop_size // 2, 2)).astype(np.int32))
                            particles[i].reset()
                            particles[i].create_kernel(img_gray, xy)
                    elif num_particles_to_delete == n_particles:
                        raise Exception("All particles deleted, object lost")
                    cv2.circle(img, tuple(coordinates_mean.astype(np.int32)),
                               2 * np.sqrt(np.sum(coordinates_std ** 2)).astype(np.int32), (0, 0, 255), 2)
                    mask += True
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
    # track_mouse_clicked_target_with_rembg(youtube_tlwh_small)
    track_mouse_clicked_target(youtube_tlwh_small)
    # track_grid(youtube_tlwh_small)
    # track_mouse_clicked_target_ORB(youtube_tlwh_large)