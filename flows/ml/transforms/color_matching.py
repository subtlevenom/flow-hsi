import copy
import cv2
import json
import numpy as np
from typing import Tuple



class ColorMatching:

    def __init__(self):
        super(ColorMatching, self).__init__()

        # Matching SURF
        self.cross_check = True
        self.feature_distance_filter = 0.9
        self.feature_mse_filter = 3
        self.surf_hessian_trashold = 125
        # install non-free opencv-contrib-python:
        # CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install -v --no-binary=opencv-contrib-python opencv-contrib-python==4
        self.detector = cv2.xfeatures2d.SURF.create()
        # self.detector = cv2.ORB.create(nfeatures=50000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=self.cross_check)

    def match_features(self, src_img, ref_img, scale = None):
        """Input RGB images"""

        if scale is not None:
            src_img = cv2.resize(src_img, (int(src_img.shape[1] / scale), int(src_img.shape[0] / scale)))
            ref_img = cv2.resize(ref_img, (int(ref_img.shape[1] / scale), int(ref_img.shape[0] / scale)))

        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

        src_keypoints, src_descriptors = self.detector.detectAndCompute(src_img, None)
        ref_keypoints, ref_descriptors = self.detector.detectAndCompute(ref_img, None)

        if self.cross_check:
            matches = self.matcher.match(src_descriptors, ref_descriptors)
            matches = sorted(matches, key = lambda x: x.distance)
            good_matches = matches[:int(len(matches)*self.feature_distance_filter)]
        else:
            matches = self.matcher.knnMatch(src_descriptors, ref_descriptors,k=2) #, self.surf_knn_match)
            # -- Filter matches using the Lowe's ratio test (set crossCheck to False)
            good_matches = []
            for m, n in matches:
                if m.distance < self.surf_ratio_thresh * n.distance:
                    good_matches.append(m)

        src_match_keypoints = np.empty((len(good_matches), 2), dtype=np.float32)
        ref_match_keypoints = np.empty((len(good_matches), 2), dtype=np.float32)

        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            src_match_keypoints[i, :] = src_keypoints[good_matches[i].queryIdx].pt
            ref_match_keypoints[i, :] = ref_keypoints[good_matches[i].trainIdx].pt

        from sklearn.datasets import make_blobs
        from sklearn import metrics
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=30, min_samples=10).fit(src_match_keypoints)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(n_clusters, n_noise)

        src_filt_points = []
        ref_filt_points = []

        for i in range(n_clusters):
            x = src_match_keypoints[labels == i]
            y = ref_match_keypoints[labels == i]

            HX, _ = cv2.findHomography(x, y, cv2.RANSAC)
            HY, _ = cv2.findHomography(y, x, cv2.RANSAC)

            # Back projection filtering
            x_proj = np.reshape(x, (1, -1, 2))
            x_proj = cv2.perspectiveTransform(x_proj, HX).astype(np.int32)
            x_proj = np.reshape(x_proj, (-1, 2))

            y_proj = np.reshape(y, (1, -1, 2))
            y_proj = cv2.perspectiveTransform(y_proj, HY).astype(np.int32)
            y_proj = np.reshape(y_proj, (-1, 2))
        
            # Back projection error
            mse = np.sum((x_proj - y)**2 + (x - y_proj)**2, axis=1)**0.5
        
            # Back projection threshold
            x_filt = x[mse <= self.feature_mse_filter]
            y_filt = y[mse <= self.feature_mse_filter]

            src_filt_points.append(x_filt)
            ref_filt_points.append(y_filt)

        return src_filt_points, ref_filt_points
    
    def extend_key_points(self, src_points:np.ndarray, ref_points:np.ndarray):
        src_ext_points = []
        ref_ext_points = []

        src_len = src_points.shape[0]
        for i in range(src_len):
            src_ext_points.append(src_points[i])
            ref_ext_points.append(ref_points[i])

        n = min(src_len, 500)

        start_indices = np.random.choice(range(src_len), n, replace=False)
        end_indices = np.random.choice(range(src_len), n, replace=False)

        for i, j in zip(start_indices, end_indices):
            if i == j:
                continue

            src_start_p = src_points[i]
            src_end_p = src_points[j]
            src_dist = np.linalg.norm(src_start_p - src_end_p)

            ref_start_p = ref_points[i]
            ref_end_p = ref_points[j]
            ref_dist = np.linalg.norm(ref_start_p - ref_end_p)

            min_dist = min(src_dist, ref_dist)
            if min_dist > 250 or min_dist < 50:
                continue

            r = int(min_dist) // 10
            src_step = (src_end_p - src_start_p) / r
            ref_step = (ref_end_p - ref_start_p) / r

            for k in range(1, r):
                src_p = src_start_p + src_step * k
                ref_p = ref_start_p + ref_step * k
                src_ext_points.append(src_p)
                ref_ext_points.append(ref_p)

        src_ext_points = np.array(src_ext_points)
        ref_ext_points = np.array(ref_ext_points)

        return src_ext_points, ref_ext_points

    def yank_colors(self, img: np.ndarray, points: np.ndarray, patch:int = None):

        img = cv2.blur(img, (21,21))

        i = points[:,1].astype(np.int32)
        j = points[:,0].astype(np.int32)

        if patch is None:
            img = cv2.blur(img, (patch,patch))

        features = img[i,j]

        return features
