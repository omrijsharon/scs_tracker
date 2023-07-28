import numpy as np
import mss
import cv2
from functools import partial

from utils.particle import Particle, ParticlesGrid, ParticleRembg
from utils.orb_tracker import ORB_tracker

from utils.helper_functions import softmax, frame_to_numpy, particles_mean_std, get_particles_attr
from utils.frame_diff_director import Director

SMALL_TOP = 160
SMALL_LEFT = 2019
SMALL_WIDTH = 1280
SMALL_HEIGHT = 720
LARGE_TOP = 80
LARGE_LEFT = 19213
LARGE_WIDTH = 1901
LARGE_HEIGHT = 1135
youtube_tlwh_small = (160, 2019, 1280, 720)
youtube_tlwh_large = (80, 1921, 1901, 1135)
uncrashed720p = (700, 1280, 1280, 720)

mouse_coords = (-1, -1)
n_particles = 11
smallest_kernel = 15
# kernel_size = np.arange(smallest_kernel//10, smallest_kernel//10 + n_particles) * 10 + 1
kernel_size = np.ones(n_particles, dtype=np.int32) * smallest_kernel
kernel_half_sizes = int((smallest_kernel-1)/2), 17
crop_size = int((kernel_half_sizes[1]*2+1) * 7)
crop_size = (crop_size//2) + 1
random_spread_inv = 128
nn_size = 3
p = 3
q = 1e-9
max_velocity = 7
particles = []
particle_max_chance_threshold = 0.001  # under this value particle will be replaced
is_random_spread = False
is_particles = False
rect_debug = False


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
                        if particle.coordinates[0] < img_gray.shape[1]*0.05 or particle.coordinates[0] >= img_gray.shape[1]*0.95 or particle.coordinates[1] < img_gray.shape[0] * 0.1 or particle.coordinates[1] >= 0.9 * img_gray.shape[0]:
                            particles2delete.append(i)
                            continue
                        xy, certainty = particle.update(img_gray)
                        if xy is None:
                            particles2delete.append(i)
                            continue
                        xy_array[i] = xy
                        if certainty * 100 > 0.4:
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
    global mouse_coords, particles, n_particles, is_particles

    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, particles, n_particles, is_particles

        if event == cv2.EVENT_LBUTTONUP:
            mouse_coords = (x, y)
            if len(particles) > 0:
                for i, particle in enumerate(particles):
                    particle.kernel_size = np.random.randint(low=kernel_half_sizes[0],high=kernel_half_sizes[1], size=(1,)).item() * 2 + 1
                    particle.reset()
                    xy = tuple(np.array(mouse_coords) + is_random_spread*np.random.randint(-crop_size//random_spread_inv, crop_size//random_spread_inv, 2))
                    particle.create_kernel((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2, xy)
                    is_particles = True
            else:
                for i in range(n_particles):
                    xy = tuple(np.array(mouse_coords) + is_random_spread*np.random.randint(-crop_size//random_spread_inv, crop_size//random_spread_inv, 2))
                    krnl_sz = np.random.randint(low=kernel_half_sizes[0],high=kernel_half_sizes[1], size=(1,)).item() * 2 + 1
                    particles.extend([Particle(krnl_sz, crop_size, nn_size, p=p, q=q, temperature=0.05, pixel_noise=0, max_velocity=max_velocity)])
                    particles[-1].reset()
                    particles[-1].create_kernel((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5) * 2, xy)
                    is_particles = True


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
        mask = np.ones((n_particles,), dtype=bool)
        max_chances = np.zeros((n_particles,))
        prev_frame = np.zeros((tlwh[3], tlwh[2]))
        # img_gray2show = np.zeros((tlwh[3], tlwh[2]), dtype=np.uint8)
        # frame_diff_2show = np.zeros((tlwh[3], tlwh[2]))
        director = Director(5, tlwh[2], tlwh[3], p, q, gradient_func="sin")
        particles_ensemble_velocity = np.zeros((2,), dtype=np.float32)
        particles_ensemble_prev_mean = np.zeros((2,), dtype=np.float32)
        temperature = 0.1
        top_bbox = 10

        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 - 0.5).astype(np.float32) * 2.0



            # hsv = director.hsv_projection(direction, magnitude)
            # cv2.imshow('hsv', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
            # cv2.imshow("magnitude", magnitude)
            # img_gray = hsv
            # img_gray = frames_diff
            # frame_diff_2show = ((frames_diff + 1) / 2 * 255).astype(np.uint8)
            # canny = cv2.Canny(frame_diff_2show, 150, 220).astype(np.uint8)
            # canny = cv2.dilate(canny, np.ones((3, 3), dtype=np.uint8), iterations=3)
            # canny = cv2.erode(canny, np.ones((3, 3), dtype=np.uint8), iterations=5)
            # contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # img_gray2show[:, :] = (1 - canny/255).astype(np.uint8)
            # # # merge img_gray2show and img into 1 frame where img_gray2show blacks are alpha-blended with img
            # img *= np.transpose(np.tile(img_gray2show, (3,1,1)), (1, 2, 0))
            # if len(contours) > 0:
            #     c = max(contours, key=cv2.contourArea)
            #     x, y, w, h = cv2.boundingRect(c)
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if is_particles:
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
                    # mask[np.argwhere(max_chances < max_chances * particle_max_chance_threshold)] = False # Old but good?
                    mask[np.argwhere(max_chances < particle_max_chance_threshold)] = False
                    particles_coordinates = get_particles_attr(particles, "coordinates")
                    coordinates_mean, coordinates_std = particles_mean_std(particles_coordinates)
                    # for each particle, check if its distance from the mean is more than 2 times the std:
                    # print(np.sqrt(np.sum(coordinates_std ** 2)))
                    particles_distances_from_mean = np.sqrt(np.sum((particles_coordinates - coordinates_mean) ** 2, axis=1))
                    mask[particles_distances_from_mean > min(2.5 * np.sqrt(np.sum(coordinates_std ** 2)), 2 * crop_size)] = False
                    coordinates_mean, coordinates_std = particles_mean_std(particles_coordinates, mask=mask)
                    # max_chances[1 - mask] = -np.inf
                    # weights = softmax(max_chances/temperature)
                    # coordinates_mean, coordinates_std = particles_mean_std(particles_coordinates, weights=weights)
                    if particles_ensemble_prev_mean[0] != 0.0:
                        particles_ensemble_velocity = coordinates_mean - particles_ensemble_prev_mean
                    particles_ensemble_prev_mean = coordinates_mean

                    num_particles_to_delete = sum(1-mask)
                    if num_particles_to_delete > 0 and num_particles_to_delete < n_particles:
                        kernel_sizes = get_particles_attr(particles, "kernel_size", mask=mask)
                        kernel_sizes_mean = np.mean(kernel_sizes)
                        kernel_sizes_std = np.std(kernel_sizes)
                        # print("kernel_sizes_mean: {}, kernel_sizes_std: {}".format(kernel_sizes_mean, kernel_sizes_std))
                        for i in np.argwhere(mask==False).flatten():
                            xy = tuple((coordinates_mean + particles_ensemble_velocity + 1.2 * np.random.randint(-crop_size // 2, crop_size // 2, 2)).astype(np.int32))
                            # particle.kernel_size = np.random.randint(low=kernel_half_sizes[0], high=kernel_half_sizes[1], size=(1,)).item() * 2 + 1
                            particle.kernel_size = np.clip(int(np.random.normal(kernel_sizes_mean, 1.*kernel_sizes_std)), a_min=kernel_half_sizes[0]*2+1, a_max=kernel_half_sizes[1]*2+1)
                            particles[i].reset()
                            particles[i].create_kernel(img_gray, xy)
                    elif num_particles_to_delete == n_particles:
                        is_particles = False
                    if is_particles:
                        # draw a cross at the mean of the particles
                        cv2.drawMarker(img, tuple(coordinates_mean.astype(np.int32)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.circle(img, tuple(coordinates_mean.astype(np.int32)),
                                   np.sqrt(2 * np.sum(coordinates_std ** 2)).astype(np.int32), (0, 0, 255), 2)
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