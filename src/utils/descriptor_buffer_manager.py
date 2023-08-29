import numpy as np
import cv2 as cv

from utils.helper_functions import match_ratio_test, kp_mean_and_std, filter_matches_by_distance, filter_top_matches


class DescriptorBuffer:
    def __init__(self, n_descriptors, descriptor_size=32):
        self.n_descriptors = n_descriptors
        self.descriptor_size = descriptor_size
        self.buffer = np.zeros((n_descriptors, descriptor_size), dtype=np.uint8)
        self.buffer_idx = 0
        self.occurrence_score = np.zeros(n_descriptors, dtype=np.uint8)
        self.recency_score = np.zeros(n_descriptors, dtype=np.uint8)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.ratio_threshold = 0.7
        self.discount_factor = 0.9
        self.occurrence_score_top_threshold = 40
        self.occurrence_score_bottom_threshold = 3
        self.matches = None

    def add(self, descriptors):
        if self.buffer_idx == 0: # first frame
            self.buffer[:len(descriptors)] = descriptors
            self.buffer_idx += len(descriptors)
            self.occurrence_score[:len(descriptors)] += 1
            self.recency_score[:len(descriptors)] += 1
            return

        if self.occurrence_score.max() > self.occurrence_score_top_threshold:
            self.clear()
        self.recency_score *= self.discount_factor

        # ratio test
        self.matches = match_ratio_test(self.matcher, descriptors, self.buffer[:self.buffer_idx], ratio_threshold=self.ratio_threshold)
        if len(self.matches) > 0:
            for match in self.matches:
                self.occurrence_score[match.trainIdx] += 1
                self.recency_score[match.trainIdx] += 1
                self.buffer[match.trainIdx] = descriptors[match.queryIdx]
            if self.buffer_idx < self.n_descriptors:  # add descriptors that were not matched to the buffer
                for i in range(len(descriptors)):
                    if i not in [m.queryIdx for m in self.matches]:
                        if self.buffer_idx < self.n_descriptors:
                            self.buffer[self.buffer_idx] = descriptors[i]
                            self.occurrence_score[self.buffer_idx] += 1
                            self.recency_score[self.buffer_idx] += 1
                            self.buffer_idx += 1
                        else:
                            break

        else: # no matches
            # add new descriptors to buffer
            if self.buffer_idx + len(descriptors) > self.n_descriptors:
                descriptors = descriptors[:self.n_descriptors - self.buffer_idx]
            self.buffer[self.buffer_idx:self.buffer_idx+len(descriptors)] = descriptors
            self.occurrence_score[self.buffer_idx:self.buffer_idx+len(descriptors)] += 1
            self.recency_score[self.buffer_idx:self.buffer_idx+len(descriptors)] += 1
            self.buffer_idx += len(descriptors)


    def clear(self):
        # sort buffer and scores by occurrence score
        sorted_idx = np.argsort(self.occurrence_score)[::-1]
        self.buffer = self.buffer[sorted_idx]
        self.occurrence_score = self.occurrence_score[sorted_idx]
        self.recency_score = self.recency_score[sorted_idx]
        # find where occurrence score is below occurrence_score_bottom_threshold and zero it.
        # since we sorted the buffer by occurrence score, we can stop when we find the first occurrence score that is below occurrence_score_bottom_threshold
        # we can also change the buffer_idx to the index of the first occurrence score that is below occurrence_score_bottom_threshold
        self.buffer_idx = np.where(self.occurrence_score < self.occurrence_score_bottom_threshold)[0][0]
        self.occurrence_score[self.buffer_idx:] = 0
        self.recency_score[self.buffer_idx:] = 0
        self.buffer[self.buffer_idx:] = 0

    def get_landmarks(self, descriptors, occurrence_score_threshold=3):
        # this functions gets descriptors and matches them to the buffer.
        # it returns the indices of the matched descriptors in the buffer with the highest occurrence score
        # if there are no matches with a high enough occurrence score, it returns None
        # ratio test
        matches = match_ratio_test(self.matcher, descriptors, self.buffer[:self.buffer_idx], ratio_threshold=self.ratio_threshold)
        if len(matches) > 0:
            # sort matches by occurrence score and return only the indices of the matches with score heighter than occurrence_score_bottom_threshold
            sorted_idx = np.argsort([self.occurrence_score[m.trainIdx] for m in matches])[::-1]
            matches = [matches[i] for i in sorted_idx if self.occurrence_score[matches[i].trainIdx] > occurrence_score_threshold]
            if len(matches) > 0:
                return np.array([m.trainIdx for m in matches])
        return None


class AdaptiveDescriptorBuffer:
    # this class gets the first descriptors. then, when getting new descriptors, it matches them to the first descriptors.
    # then, it changes the first descriptors to the new descriptors if the matches are good enough.
    # by this, it maintains the original descriptors, but adapts them to the current scene.
    def __init__(self, matcher, n_descriptors, min_n_matches, ratio_threshold=0.7, descriptor_size=32):
        self.n_descriptors = n_descriptors
        self.descriptor_size = descriptor_size
        self.min_n_matches = min_n_matches
        self.buffer = np.zeros((n_descriptors, descriptor_size), dtype=np.uint8)
        self.matcher = matcher
        self.ratio_threshold = ratio_threshold
        self.matches = None
        self.mean = None
        self.std = None
        self.kp = None

    def clear(self):
        self.buffer = np.zeros((self.n_descriptors, self.descriptor_size), dtype=np.uint8)

    def reset(self, xy, keypoints, descriptors):
        # sort the keypoints and descriptors by their euclidean distance from xy.
        # then take the first n_descriptors that are closest to xy
        # this way, we get the descriptors that are closest to xy
        kp_coords = np.array([k.pt for k in keypoints])
        kp_distances = np.linalg.norm(kp_coords - xy, axis=1)
        sorted_idx = np.argsort(kp_distances)
        self.buffer = descriptors[sorted_idx[:self.n_descriptors]]
        self.mean, self.std = kp_mean_and_std(np.array(keypoints)[sorted_idx[:self.n_descriptors]])
        # return the distances of the n_descriptors closest keypoints to xy
        return kp_distances[sorted_idx[:self.n_descriptors]]

    def update(self, keypoints, descriptors):
        # ratio test
        self.matches = self.try_to_match(descriptors) # for orb
        # self.matches = self.matcher.match(descriptors, self.buffer) # for sift
        if self.matches is not None:
            for match in self.matches:
                self.buffer[match.trainIdx] = descriptors[match.queryIdx]
            queryIdx = [m.queryIdx for m in self.matches]
            print("len queryIdx:", len(queryIdx))
            self.kp = list(np.array(keypoints)[queryIdx])
            self.mean, self.std = kp_mean_and_std(self.kp)
            return True
        return False

    def try_to_match(self, descriptors):
        len_matches = 0
        ratio_threshold = self.ratio_threshold
        while len_matches < self.min_n_matches and ratio_threshold < 0.9:
            matches = match_ratio_test(self.matcher, descriptors, self.buffer, ratio_threshold=ratio_threshold)
            # queryIdx = [m.queryIdx for m in matches]
            # trainIdx = [m.trainIdx for m in matches]
            # print("len queryIdx:", len(queryIdx), "         len trainIdx:", len(trainIdx))
            matches = filter_top_matches(matches, n_top_matches=self.min_n_matches)
            matches = filter_matches_by_distance(matches, max_distance=75)
            len_matches = len(matches)
            if len_matches >= self.min_n_matches:
                return matches
            ratio_threshold = (1 + ratio_threshold) / 2
            print("ratio_threshold:", ratio_threshold)
        return None
