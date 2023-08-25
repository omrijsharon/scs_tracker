import numpy as np
import cv2 as cv

from utils.helper_functions import match_ratio_test


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