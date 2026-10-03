#this code finds the board without using any Neural network models. It works from the board's grid lines, so pieces standing on it don't break the detection.


import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import chess
import chess.engine
import time
import sys
from move_detector import MoveTracker

INTERVAL = 0.5  # seconds between captures; moves are only read once the board is steady for a few captures
LAST_CAPTURE_TIME = 0
# How checkerboard-like the found 8x8 area has to be (1.0 = perfect, ~0 = random)
MIN_CHECKER_SCORE = 0.25

MIN_BOARD_LENGTH = 240
MAX_BOARD_LENGTH = 400
# Class id to label mapping must match training
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}


def find_lines(edges):
    # The vote threshold scales with the frame instead of a fixed number, and
    # pieces only cost a line some votes, they can't cut it in half like a contour
    h, w = edges.shape
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=int(0.2 * min(h, w)))
    if lines is None:
        return None
    return lines[:400, 0]  # HoughLines returns the strongest lines first


def angle_diff(a, b):
    # difference between two line directions in degrees, 0..90 (180 deg = same line)
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


def direction_pairs(lines, spread=30, peaks=4):
    # Board lines come in two main directions, but with strong perspective one
    # of them fans out and piece edges add diagonal junk. So take the few most
    # common directions and return every plausible pair; the checkerboard
    # test later decides which pair is the board
    angles = np.degrees(lines[:, 1]) % 180
    hist = np.bincount(angles.astype(int) % 180, minlength=180).astype(float)
    hist = np.convolve(np.concatenate([hist[-10:], hist, hist[:10]]), np.ones(21), 'valid')
    tops = []
    for _ in range(peaks):
        if hist.max() <= 0:
            break
        top = int(np.argmax(hist))
        tops.append(top)
        hist[angle_diff(np.arange(180), top) < 30] = 0
    pairs = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            if angle_diff(tops[i], tops[j]) < 40:
                continue
            to_i, to_j = angle_diff(angles, tops[i]), angle_diff(angles, tops[j])
            a = lines[(to_i < spread) & (to_i <= to_j)]
            b = lines[(to_j < spread) & (to_j < to_i)]
            if len(a) >= 2 and len(b) >= 2:
                pairs.append((a, b))
    return pairs


def merge_close_lines(lines, center, min_gap, keep=25):
    # Flip every normal to point the same way, measure each line's signed
    # distance from the image center, and drop near-duplicates of stronger lines
    mean_angle = 0.5 * math.atan2(np.sin(2 * lines[:, 1]).mean(), np.cos(2 * lines[:, 1]).mean())
    kept = []  # (position, theta, rho)
    for rho, theta in lines:
        if math.cos(theta - mean_angle) < 0:
            rho, theta = -rho, theta - np.pi
        pos = rho - (center[0] * math.cos(theta) + center[1] * math.sin(theta))
        if any(abs(pos - p) < min_gap and abs(theta - t) < np.radians(5) for p, t, _ in kept):
            continue
        kept.append((pos, theta, rho))
        if len(kept) == keep:
            break
    kept.sort()
    return np.array([(rho, theta) for _, theta, rho in kept])


def intersections(lines_a, lines_b):
    # points[i, j] = where line i of the first direction meets line j of the second
    points = np.full((len(lines_a), len(lines_b), 2), np.nan)
    for i, (r1, t1) in enumerate(lines_a):
        for j, (r2, t2) in enumerate(lines_b):
            m = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
            if abs(np.linalg.det(m)) > 1e-6:
                points[i, j] = np.linalg.solve(m, [r1, r2])
    return points


def lattice_inliers(H, points, tol=0.15):
    # Map points into "board units" and keep the ones that land on whole numbers
    uv = cv.perspectiveTransform(points.reshape(-1, 1, 2), H).reshape(-1, 2)
    grid = np.round(uv)
    ok = np.all(np.abs(uv - grid) < tol, axis=1) & np.all(np.abs(grid) < 20, axis=1)
    return ok, grid


def fit_grid(points, img_shape):
    # Try every pair of neighbouring lines as "one square" and see how many
    # line crossings that puts on a regular grid. Returns the candidate grids,
    # most crossings first
    h, w = img_shape[:2]
    flat = points.reshape(-1, 2)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    flat = flat[(flat[:, 0] > -w) & (flat[:, 0] < 2 * w) & (flat[:, 1] > -h) & (flat[:, 1] < 2 * h)].astype(np.float32)
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    min_area = (0.02 * min(h, w)) ** 2

    candidates = {}  # inlier set -> (count, H, grid points)
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            quad = np.float32([points[i, j], points[i + 1, j], points[i + 1, j + 1], points[i, j + 1]])
            if not np.all(np.isfinite(quad)) or cv.contourArea(quad) < min_area:
                continue
            H = cv.getPerspectiveTransform(quad, unit)
            for _ in range(3):
                ok, grid = lattice_inliers(H, flat)
                if ok.sum() < 4:
                    break
                H, _ = cv.findHomography(flat[ok], grid[ok].astype(np.float32))
                if H is None:
                    break
            if H is None:
                continue
            ok, grid = lattice_inliers(H, flat)
            count = len(np.unique(grid[ok], axis=0))
            if count >= 16:
                candidates[ok.tobytes()] = (count, H, grid[ok])
    # the same grid is found from many starting squares, keep each one once
    return sorted(candidates.values(), key=lambda c: -c[0])


def pick_8x8(img, H, grid_pts, px=16):
    # The grid may be missing its outer lines (hidden by pieces) or include
    # extra ones (board frame), so slide an 8x8 window over it and keep the
    # spot that holds the most squares of the right colour
    xmin, ymin = grid_pts.min(axis=0).astype(int)
    xmax, ymax = grid_pts.max(axis=0).astype(int)
    x_lo, x_hi = min(xmin, xmax - 8), max(xmax, xmin + 8)
    y_lo, y_hi = min(ymin, ymax - 8), max(ymax, ymin + 8)
    nx, ny = x_hi - x_lo, y_hi - y_lo

    to_px = np.array([[px, 0, -x_lo * px], [0, px, -y_lo * px], [0, 0, 1]], dtype=np.float64)
    M = to_px @ H
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    flat = cv.warpPerspective(lab, M, (nx * px, ny * px)).astype(np.float32)
    inside = cv.warpPerspective(np.ones(img.shape[:2], np.uint8), M, (nx * px, ny * px), flags=cv.INTER_NEAREST)
    m = px // 6  # ignore the cell borders
    cells = flat.reshape(ny, px, nx, px, 3)[:, m:-m, :, m:-m].transpose(0, 2, 1, 3, 4)  # (row, col, y, x, colour)
    cells_inside = inside.reshape(ny, px, nx, px)[:, m:-m, :, m:-m].mean(axis=(1, 3)) >= 0.9

    # The two square colours: the median colour of each half of the checker
    # pattern, taken between the grid lines that were actually found
    parity = np.add.outer(np.arange(ny) + y_lo, np.arange(nx) + x_lo) % 2
    core = np.zeros((ny, nx), bool)
    core[ymin - y_lo:ymax - y_lo, xmin - x_lo:xmax - x_lo] = True
    core &= cells_inside
    if not (core & (parity == 0)).any() or not (core & (parity == 1)).any():
        return None, 0
    colour0 = np.median(cells[core & (parity == 0)].reshape(-1, 3), axis=0)
    colour1 = np.median(cells[core & (parity == 1)].reshape(-1, 3), axis=0)
    gap = np.linalg.norm(colour0 - colour1)
    if gap < 15:
        return None, 0  # both halves look the same: not a checkerboard

    # For every square: share of pixels in the colour it should have minus
    # share in the other one. Pieces match neither colour, so a square that is
    # half hidden by a piece still counts; the frame and table score ~0
    near0 = (np.linalg.norm(cells - colour0, axis=-1) < gap / 2).mean(axis=(2, 3))
    near1 = (np.linalg.norm(cells - colour1, axis=-1) < gap / 2).mean(axis=(2, 3))
    fits = np.where(parity == 0, near0 - near1, near1 - near0)

    best_score, best_xy = -1, None
    for y0 in range(ny - 7):
        for x0 in range(nx - 7):
            if not cells_inside[y0:y0 + 8, x0:x0 + 8].all():
                continue
            score = fits[y0:y0 + 8, x0:x0 + 8].mean()
            if score > best_score:
                best_score, best_xy = score, (x0 + x_lo, y0 + y_lo)
    return best_xy, best_score


def order_corners(corners):
    # top-left, top-right, bottom-right, bottom-left (clockwise on screen)
    c = corners - corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(c[:, 1], c[:, 0]))]
    start = np.argmin(corners.sum(axis=1))
    return np.roll(corners, -start, axis=0).astype(np.float32)


def find_board_corners(img, edges, debug=None):
    # Returns the 4 corners of the 8x8 playing area (not the board frame) or None
    lines = find_lines(edges)
    if lines is None or len(lines) < 4:
        return None, 'not enough lines'
    center = (img.shape[1] / 2, img.shape[0] / 2)
    min_gap = max(4, 0.015 * min(img.shape[:2]))

    # A grid with half-size squares or one built from diagonals can have just
    # as many crossings as the real one, but it won't look like a checkerboard
    best = (-1, None, None, 0, None)
    for a, b in direction_pairs(lines):
        a = merge_close_lines(a, center, min_gap)
        b = merge_close_lines(b, center, min_gap)
        for count, H, grid_pts in fit_grid(intersections(a, b), img.shape)[:10]:
            xy, score = pick_8x8(img, H, grid_pts)
            if xy is not None and score > best[0]:
                best = (score, H, xy, count, (a, b))
    score, H, xy, count, best_lines = best
    if xy is None:
        return None, 'no grid found'

    if debug is not None:
        for rho, theta in np.vstack(best_lines):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            cv.line(debug, (int(x0 - 3000 * b), int(y0 + 3000 * a)), (int(x0 + 3000 * b), int(y0 - 3000 * a)), (0, 255, 255), 1)

    if score < MIN_CHECKER_SCORE:
        return None, f'no checkerboard pattern (board score {score:.2f})'
    x0, y0 = xy
    board = np.float32([[x0, y0], [x0 + 8, y0], [x0 + 8, y0 + 8], [x0, y0 + 8]])
    corners = cv.perspectiveTransform(board.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    return order_corners(corners), f'{count} line crossings, board score {score:.2f}'


def detect_chessboard(frame):    
        img = frame.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise from pieces
        blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
        
        #Detecting edges
        canny_img = cv.Canny(blurred, 50, 100)
        cv.imshow('edges', canny_img)

        # Find the board from its grid lines instead of its outer contour:
        # pieces on the edge ranks stick out over the border and merge with it,
        # but they can't hide a whole grid line
        board_corners, info = find_board_corners(frame, canny_img, debug=img)
        cv.imshow('Lines', img)

        is_board = board_corners is not None

        # we divide the board into 64 pieces with angle consideration
        if is_board == True:
            new_board = frame.copy()
            
            # Calculate the size of the board in the transformed space
            # We'll use a standard 8x8 grid size for the transformed image
            grid_size = 400  # Size of the transformed board image
            square_size = grid_size // 8
            
            # Define the destination corners for perspective transform
            dst_corners = np.array([
                [0, 0],                    # Top-left
                [grid_size, 0],            # Top-right
                [grid_size, grid_size],    # Bottom-right
                [0, grid_size]             # Bottom-left
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            perspective_matrix = cv.getPerspectiveTransform(board_corners, dst_corners)
            
            # Apply perspective transform to get the straightened board
            transformed_board = cv.warpPerspective(new_board, perspective_matrix, (grid_size, grid_size))
            
            # Draw the detected board corners on the original image
            for i, corner in enumerate(board_corners):
                cv.circle(new_board, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
                cv.putText(new_board, str(i), tuple(corner.astype(int)), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create 8x8 grid of squares on the transformed board
            squares_grid = []
            for row in range(8):
                for col in range(8):
                    # Calculate square coordinates in transformed space
                    x1 = col * square_size
                    y1 = row * square_size
                    x2 = (col + 1) * square_size
                    y2 = (row + 1) * square_size
                    
                    # Extract square from the transformed board
                    square = transformed_board[y1:y2, x1:x2]
                    
                    # Calculate the corresponding coordinates in the original image
                    # Transform the square corners back to original image space
                    square_corners_transformed = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)
                    
                    # Apply inverse perspective transform
                    inverse_matrix = cv.getPerspectiveTransform(dst_corners, board_corners)
                    square_corners_original = cv.perspectiveTransform(
                        square_corners_transformed.reshape(-1, 1, 2), inverse_matrix
                    ).reshape(-1, 2)
                    
                    # Draw the transformed square on the original image
                    square_corners_original_int = square_corners_original.astype(int)
                    cv.polylines(new_board, [square_corners_original_int], True, (0, 255, 0), 2)
                    
                    # Add square coordinates and content to the grid
                    squares_grid.append({
                        'position': (row, col),
                        'coordinates_transformed': (x1, y1, x2, y2),
                        'coordinates_original': square_corners_original.tolist(),
                        'content': square
                    })
                
            # Display both the original image with drawn squares and the transformed board
            cv.imshow('Divided Chess Board (Original)', new_board)
            cv.imshow('Transformed Chess Board', transformed_board)

            return squares_grid
        else:
            print(f"no chessboard detected ({info}), trying again")
            return None

if __name__ == '__main__':
    # Camera number (default 0), a video file or a stream URL:
    #   python board_finder.py
    #   python board_finder.py game.mp4
    #   python board_finder.py http://192.168.1.20:8080/video
    source = sys.argv[1] if len(sys.argv) > 1 else '0'
    if source.isdigit():
        # DirectShow only exists on Windows
        camera = cv.VideoCapture(int(source), cv.CAP_DSHOW if sys.platform == 'win32' else cv.CAP_ANY)
    else:
        camera = cv.VideoCapture(source)
    if not camera.isOpened():
        print("Не удалось открыть камеру")
        exit()

    tracker = MoveTracker()
    print("Set up the starting position. Keys (in a camera window): q = quit, r = new game,")
    print("u = take back a wrongly recognised move, m = type the move that was played")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Не удалось считать кадр")
            break

        now = time.time()
        if now - LAST_CAPTURE_TIME >= INTERVAL:
            cv.imshow('Screenshot',frame)
            tracker.update(detect_chessboard(frame), frame)
            LAST_CAPTURE_TIME = now

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            print("New game: set up the starting position")
        if key == ord('u'):
            tracker.undo()
        if key == ord('m'):
            text = input("Move that was played (e.g. Nf3), Enter to cancel: ").strip()
            if text:
                tracker.play(text)

    camera.release()
    cv.destroyAllWindows()


