# Keeps track of where the board is in the camera picture.
# Finding a board from scratch is hard when it is full of pieces (squares are
# hidden, rows of pieces look like extra lines), but easy when it is empty. So
# the board is found once and "locked", ideally while it is still empty, and
# after that it is followed from picture to picture as the camera (your head)
# moves, instead of being searched for again:
#   1. guess where it moved: match image features near the board between the
#      last picture where the board was seen well and the new one
#   2. snap the board's 9 + 9 lines onto the edges seen near that guess

import cv2 as cv
import numpy as np

LOCK_CAPTURES = 3     # the board must be found in the same place this many captures in a row to lock
LOCK_TOLERANCE = 0.3  # "the same place": every corner within this many squares
SNAP_DISTANCE = 0.25  # an edge counts towards a board line if it lies within this many squares of it
MIN_LINES = 3         # board lines needed in each direction to fit the board again
MAX_JUMP = 0.4        # the fit may move a corner at most this many squares away from the guess
EMPTY_SCORE = 0.85    # a board found with this board score (or more) has no pieces on it
SEARCH_AGAIN = 10     # after this many captures in a row with the board lost, look for it from scratch

BOARD = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])


def edges_of(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray, cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 50, 100)


def square_size(corners):
    # rough size of a square in pixels
    return np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean() / 8


def around_board(shape, corners, margin):
    # mask of the board plus `margin` squares around it (frame and table)
    H = cv.getPerspectiveTransform(BOARD, np.float32(corners))
    outline = np.float32([[-margin, -margin], [8 + margin, -margin], [8 + margin, 8 + margin], [-margin, 8 + margin]])
    poly = cv.perspectiveTransform(outline.reshape(-1, 1, 2), H).reshape(-1, 2)
    mask = np.zeros(shape[:2], np.uint8)
    cv.fillConvexPoly(mask, poly.astype(np.int32), 255)
    return mask


def top_view(gray, corners, px=12):
    # The board plus one square of frame and table around it, seen from above
    # (blurred, so a few pixels of difference don't matter), and which part of
    # that view was inside the picture
    H = cv.getPerspectiveTransform(BOARD, np.float32(corners))
    M = np.array([[px, 0, px], [0, px, px], [0, 0, 1.0]]) @ np.linalg.inv(H)
    n = 10 * px
    view = cv.warpPerspective(gray, M, (n, n)).astype(np.float32)
    inside = cv.warpPerspective(np.ones_like(gray), M, (n, n), flags=cv.INTER_NEAREST) > 0
    return cv.GaussianBlur(view, (5, 5), 0), inside


def likeness(a, a_inside, b, b_inside):
    # how alike two views are (correlation, 1 = identical), over the part both show
    both = a_inside & b_inside
    if both.mean() < 0.3:
        return -1.0
    x, y = a[both] - a[both].mean(), b[both] - b[both].mean()
    return float((x * y).mean() / (x.std() * y.std() + 1e-6))


class BoardTracker:
    def __init__(self, find_board, find_segments):
        # find_board(frame, edges) -> (corners or None, info) finds a board from
        # scratch; find_segments(edges) -> straight edge pieces (x1, y1, x2, y2)
        self.find_board, self.find_segments = find_board, find_segments
        self.orb = cv.ORB_create(1500)
        self.reset()

    def reset(self):
        self.state = 'searching'  # searching -> locked
        self.seen = []            # recent sightings while searching
        self.corners = None       # where the board is now
        self.anchor = None        # (gray, corners) of the last picture where the board was seen well
        self.empty = False        # was the board empty when it was locked?
        self.lost = 0             # captures in a row with the board lost
        self.previous = None      # corners in the previous capture (None if the board wasn't given then)
        self.status = 'Looking for the board: point the camera at the whole board (empty is best)'

    def update(self, frame):
        # Returns the board's 4 corners in this frame, or None when it can't be
        # trusted (not locked yet, or out of view)
        gray, edges = edges_of(frame)
        corners = self.follow(frame, gray, edges)
        self.previous = corners
        return corners

    def follow(self, frame, gray, edges):
        if self.state == 'searching':
            if self.anchor is not None:
                # it was locked before: maybe the camera just looks at it again
                guess = self.guess(gray)
                fitted = self.snap(edges, guess) if guess is not None else None
                if fitted is not None and self.trust(gray, fitted):
                    self.state, self.seen, self.lost, self.empty = 'locked', [], 0, False
                    self.anchor = (gray, fitted)
                    self.status = 'Board locked'
                    print('Board found again')
                    return fitted
            return self.search(frame, gray, edges)

        guess = self.guess(gray)
        if guess is None:
            self.lost += 1
            self.status = 'Board lost: look at the board again'
            if self.lost >= SEARCH_AGAIN:
                # looked away too long (or the scene changed): find it from scratch
                self.state, self.seen, self.lost = 'searching', [], 0
            return None
        self.lost = 0
        fitted = self.snap(edges, guess)
        if fitted is None:
            # not enough board lines visible (hand over the board, motion
            # blur, looking away...): the guess alone can't be trusted, so
            # report nothing this time
            self.status = 'Board partly hidden'
            return None
        if not self.trust(gray, fitted):
            self.status = 'Board partly hidden'
            return None
        self.corners = fitted
        self.anchor = (gray, fitted)
        self.status = 'Board locked'
        return fitted

    def search(self, frame, gray, edges):
        corners, info = self.find_board(frame, edges)
        if corners is None:
            self.seen = []
            self.status = f'Looking for the board: point the camera at the whole board (empty is best) [{info}]'
            return None
        if self.seen:
            # the camera (your head) moves a little between captures, so
            # compare with where the last sighting should be now
            last_gray, last_corners = self.seen[-1][:2]
            expected = self.guess(gray, (last_gray, last_corners))
            if expected is None:
                expected = last_corners
            if np.abs(corners - expected).max() > LOCK_TOLERANCE * square_size(corners):
                self.seen = []  # not the same board position: start counting again
        self.seen.append((gray, corners, info))
        if len(self.seen) < LOCK_CAPTURES:
            self.status = 'Board found, hold still...'
            return None
        self.state = 'locked'
        self.corners = corners
        self.anchor = (gray, corners)
        score = float(info.split('board score ')[-1].rstrip(')')) if 'board score' in info else 0
        self.seen = []
        self.empty = score >= EMPTY_SCORE
        self.status = 'Board locked'
        print('Board locked' + (' (empty): set up the pieces' if self.empty else ''))
        return corners

    def guess(self, gray, anchor=None):
        # Where did the board move since the last good picture? Match image
        # features on and around the board (frame, table, pieces) and fit one
        # perspective change to them, ignoring matches that disagree (hands,
        # moved pieces, people in the background)
        old_gray, old_corners = anchor if anchor is not None else self.anchor
        mask = around_board(old_gray.shape, old_corners, 1.5)
        kp1, des1 = self.orb.detectAndCompute(old_gray, mask)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None
        matches = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True).match(des1, des2)
        if len(matches) < 12:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst = np.float32([kp2[m.trainIdx].pt for m in matches])
        M, inliers = cv.findHomography(src, dst, cv.RANSAC, 4.0)
        if M is None or inliers.sum() < 12:
            return None
        return cv.perspectiveTransform(old_corners.reshape(-1, 1, 2), M).reshape(-1, 2)

    def trust(self, gray, corners):
        # The squares repeat, so the board can line up a whole square or two off
        # and still fit the lines perfectly. That's impossible if it barely moved
        # since the previous capture. Otherwise compare with the last good view:
        # the board must look more like it at this position than shifted by one
        # or two squares (the frame, table and pieces only line up at the right one)
        if self.previous is not None and np.abs(corners - self.previous).max() < 0.5 * square_size(corners):
            return True
        old_gray, old_corners = self.anchor
        ref, ref_inside = top_view(old_gray, old_corners)
        H = cv.getPerspectiveTransform(BOARD, np.float32(corners))
        here = likeness(ref, ref_inside, *top_view(gray, corners))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            shifted = cv.perspectiveTransform((BOARD + [dx, dy]).reshape(-1, 1, 2), H).reshape(-1, 2)
            if likeness(ref, ref_inside, *top_view(gray, shifted)) > here - 0.05:
                return False
        return here > 0.3

    def snap(self, edges, corners):
        # Fit the board to the edges seen in the picture: every straight edge
        # piece that runs along one of the board's 9 + 9 lines (as placed by the
        # guess) counts for that line; each line is fitted through its edge
        # pieces, and the board through the crossings of those lines
        segments = self.find_segments(edges)
        if segments is None:
            return None
        guess = corners
        for _ in range(2):  # a second pass with the improved board catches more edges
            to_board = cv.getPerspectiveTransform(np.float32(corners), BOARD)
            ends = cv.perspectiveTransform(segments.reshape(-1, 1, 2).astype(np.float32), to_board).reshape(-1, 2, 2)
            lines = {}
            for axis in (0, 1):  # 0: lines x = k (one direction), 1: lines y = k (the other)
                along = 1 - axis
                for (b1, b2), seg in zip(ends, segments):
                    k = int(round((b1[axis] + b2[axis]) / 2))
                    if not 0 <= k <= 8:
                        continue
                    if abs(b1[axis] - k) > SNAP_DISTANCE or abs(b2[axis] - k) > SNAP_DISTANCE:
                        continue
                    if abs(b2[along] - b1[along]) < 0.3 or max(b1[along], b2[along]) < -0.3 or min(b1[along], b2[along]) > 8.3:
                        continue  # too short to tell the direction, or off the board
                    lines.setdefault((axis, k), []).append(seg)
            fitted = {}
            for key, segs in lines.items():
                pts = np.float32(segs).reshape(-1, 2)
                vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_HUBER, 0, 0.01, 0.01).ravel()
                fitted[key] = np.cross([x0, y0, 1], [x0 + vx, y0 + vy, 1])
            xs = [k for axis, k in fitted if axis == 0]
            ys = [k for axis, k in fitted if axis == 1]
            if len(xs) < MIN_LINES or len(ys) < MIN_LINES:
                return None
            board_pts, image_pts = [], []
            for i in xs:
                for j in ys:
                    p = np.cross(fitted[(0, i)], fitted[(1, j)])
                    if abs(p[2]) > 1e-9:
                        board_pts.append([i, j])
                        image_pts.append(p[:2] / p[2])
            H, _ = cv.findHomography(np.float32(board_pts), np.float32(image_pts), cv.RANSAC, 0.1 * square_size(corners) + 1)
            if H is None:
                return None
            corners = cv.perspectiveTransform(BOARD.reshape(-1, 1, 2), H).reshape(-1, 2)
            if np.abs(corners - guess).max() > MAX_JUMP * square_size(guess):
                return None  # far from the guess: probably snapped onto the wrong lines
        return corners
