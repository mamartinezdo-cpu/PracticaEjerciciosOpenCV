import cv2
import numpy as np
from collections import namedtuple


class PoseEstimator:
    def __init__(self):
        # FLANN parameters para ORB
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)

        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image rect keypoints descriptors data')
        self.tracked_target = namedtuple('Tracked', 'target points_prev points_cur H quad')

        # ORB: rápido y gratuito
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    def add_target(self, image, rect, data=None):
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []

        for kp, desc in zip(*self.detect_features(image)):
            x, y = kp.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(kp)
                descriptors.append(desc)

        if not descriptors:
            print("⚠️ No se detectaron características en el ROI seleccionado.")
            return

        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints, descriptors=descriptors, data=data)
        self.tracking_targets.append(target)

    def track_target(self, frame):
        keypoints, descriptors = self.detect_features(frame)
        if len(keypoints) < self.min_matches:
            return []

        try:
            matches = self.feature_matcher.knnMatch(descriptors, k=2)
        except Exception:
            return []

        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(good_matches) < self.min_matches:
            return []

        matches_by_target = [[] for _ in range(len(self.tracking_targets))]
        for match in good_matches:
            matches_by_target[match.imgIdx].append(match)

        tracked = []
        for idx, matches in enumerate(matches_by_target):
            if len(matches) < self.min_matches:
                continue

            target = self.tracking_targets[idx]
            pts_prev = np.float32([target.keypoints[m.trainIdx].pt for m in matches])
            pts_cur = np.float32([keypoints[m.queryIdx].pt for m in matches])

            H, status = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3.0)
            if H is None:
                continue

            status = status.ravel() != 0
            pts_prev, pts_cur = pts_prev[status], pts_cur[status]

            if len(pts_prev) < self.min_matches:
                continue

            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([[x_start, y_start], [x_end, y_start],
                               [x_end, y_end], [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            tracked.append(self.tracked_target(target, pts_prev, pts_cur, H, quad))

        tracked.sort(key=lambda x: len(x.points_prev), reverse=True)
        return tracked

    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    def clear_targets(self):
        self.feature_matcher.clear()
        self.tracking_targets = []


class ROISelector:
    def __init__(self, win_name, init_frame, callback_func):
        self.callback_func = callback_func
        self.win_name = win_name
        self.drag_start = None
        self.rect = None
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_event, init_frame)

    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.rect = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self.callback_func(self.rect)
            self.drag_start = None

    def draw_rect(self, img, rect):
        if not rect:
            return
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
