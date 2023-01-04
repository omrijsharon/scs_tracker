from utils.scs import scs_filter


def image_processing(frame, kernel, ones_kernel, p=3, q=1e-6):
    return scs_filter(frame, kernel, ones_kernel, p, q)