#this code finds the board without using any Neural network models. It works from the board's grid lines, so pieces standing on it don't break the detection.


import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import chess
import chess.engine
import time
import sys
import os
from move_detector import MoveTracker
from board_tracker import BoardTracker

INTERVAL = 0.5  # seconds between captures; moves are only read once the board is steady for a few captures
FOLLOW_INTERVAL = 0.1  # the board is followed more often, so it moves little between looks
LAST_CAPTURE_TIME = 0
# How checkerboard-like the found 8x8 area has to be (1.0 = perfect, ~0 = random)
MIN_CHECKER_SCORE = 0.37
# How much of the grid's lines must lie on edges in the photo
MIN_LINES_ON_EDGES = 0.75

MIN_BOARD_LENGTH = 240
MAX_BOARD_LENGTH = 400
# Class id to label mapping must match training
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}


def find_segments(edges):
    # Straight stretches of edge as (x1, y1, x2, y2)
    h, w = edges.shape
    segments = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=max(10, int(0.04 * min(h, w))), maxLineGap=3)
    return None if segments is None else segments.reshape(-1, 4).astype(np.float64)


def vanishing_points(segments, count=3, max_angle=2.0):
    # In a photo the lines of each board direction meet in one point (a
    # vanishing point; far away, or "at infinity" when they look parallel).
    # Seen from a seated player the files fan out over ~90 degrees, so
    # grouping lines by angle fails, but they still all point at one spot.
    # Try the crossing of pairs of long segments as that spot, and keep the
    # spots the most segment length points at. A segment only counts if the
    # spot lies beyond its ends, so places where grid lines cross don't count.
    # Returns [(point, member mask)], point homogeneous (x, y, 1) or (dx, dy, 0)
    p1, p2 = segments[:, :2], segments[:, 2:]
    length = np.linalg.norm(p2 - p1, axis=1)
    mid, d = (p1 + p2) / 2, (p2 - p1) / length[:, None]
    lines = np.cross(np.column_stack([p1, np.ones(len(p1))]), np.column_stack([p2, np.ones(len(p2))]))

    longest = np.argsort(-length)[:80]
    i, j = np.triu_indices(len(longest), 1)
    cands = np.cross(lines[longest[i]], lines[longest[j]])
    cands = cands[np.linalg.norm(cands[:, :2], axis=1) > 1e-9]
    if len(cands) == 0:
        return []
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)

    def support(v):
        # for each segment (rows) and point (columns): does the segment point
        # at the point, with the point beyond the segment's ends?
        u = v[None, :, :2] - mid[:, None, :] * v[None, :, 2:3]  # direction from the segment to v (scaled)
        sin = np.abs(d[:, None, 0] * u[..., 1] - d[:, None, 1] * u[..., 0]) / (np.linalg.norm(u, axis=2) + 1e-12)
        far = np.abs(v[None, :, 2])
        along = np.abs((u * d[:, None, :]).sum(axis=2)) / np.maximum(far, 1e-12)
        beyond = (far < 1e-9) | (along > length[:, None] / 2 + 2)
        return (sin < np.sin(np.radians(max_angle))) & beyond

    points_at = support(cands)
    used = np.zeros(len(segments), bool)
    found = []
    for _ in range(count):
        scores = (points_at[~used] * length[~used, None]).sum(axis=0)
        best = int(np.argmax(scores))
        best_members = points_at[:, best] & ~used
        if best_members.sum() < 4:
            break
        # refit the point to all its segments: the point closest to all their lines
        n = lines[best_members] / np.linalg.norm(lines[best_members][:, :2], axis=1, keepdims=True)
        v = np.linalg.eigh((n * length[best_members, None]).T @ n)[1][:, 0]
        members = support(v[None, :])[:, 0] & ~used
        found.append((v, members))
        used |= members
    return found


def to_rho_theta(line):
    # homogeneous line (a, b, c) -> (rho, theta) as HoughLines gives them
    a, b, c = line / np.linalg.norm(line[:2])
    theta, rho = np.arctan2(b, a), -c
    if theta < 0:
        theta, rho = theta + np.pi, -rho
    return rho, theta


def snapped_lines(segments, members, point, reference, middle, min_gap, keep=25):
    # The board lines of one direction: every segment of the set is turned into
    # the line through the vanishing point and the segment's middle, and lines
    # that cross the reference line within min_gap pixels of each other are
    # merged (weighted by segment length). Returns (rho, theta) sorted across the board
    seg = segments[members]
    length = np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1)
    mids = np.column_stack([(seg[:, :2] + seg[:, 2:]) / 2, np.ones(len(seg))])
    crossings = np.cross(np.cross(point[None, :], mids), reference[None, :])
    ok = np.abs(crossings[:, 2]) > 1e-9
    direction = np.array([reference[1], -reference[0]]) / np.linalg.norm(reference[:2])
    pos = (crossings[ok, :2] / crossings[ok, 2:3] - middle[:2]) @ direction
    length = length[ok]
    order = np.argsort(pos)
    groups = []  # [sum of length * pos, sum of length]
    last = None
    for k in order:
        if last is not None and pos[k] - last < min_gap:
            groups[-1][0] += length[k] * pos[k]
            groups[-1][1] += length[k]
        else:
            groups.append([length[k] * pos[k], length[k]])
        last = pos[k]
    groups = sorted(groups, key=lambda g: -g[1])[:keep]
    spots = sorted(g[0] / g[1] for g in groups)
    lines = [to_rho_theta(np.cross(point, np.append(middle[:2] + t * direction, 1.0))) for t in spots]
    return np.array(lines)


def family_pairs(edges, min_gap):
    # Pairs of board-line sets, one per vanishing point, each sorted across the board
    segments = find_segments(edges)
    if segments is None or len(segments) < 8:
        return []
    points = vanishing_points(segments)
    pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            (va, ma), (vb, mb) = points[i], points[j]
            both = segments[ma | mb]
            middle = np.append(np.median((both[:, :2] + both[:, 2:]) / 2, axis=0), 1.0)
            # sort each set along the line of the other set through the middle
            a = snapped_lines(segments, ma, va, np.cross(middle, vb), middle, min_gap)
            b = snapped_lines(segments, mb, vb, np.cross(middle, va), middle, min_gap)
            if len(a) >= 2 and len(b) >= 2:
                pairs.append((a, b))
    return pairs


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
                candidates[ok.tobytes()] = (count, H, grid[ok], flat[ok])
    # the same grid is found from many starting squares, keep each one once
    return sorted(candidates.values(), key=lambda c: -c[0])


def pick_8x8(img, H, grid_pts, px=16):
    # The grid may be missing its outer lines (hidden by pieces) or include
    # extra ones (board frame, table edge, lines in the background), so slide an
    # 8x8 window over it and keep the spot that holds the most squares of the
    # right colour
    xmin, ymin = grid_pts.min(axis=0).astype(int)
    xmax, ymax = grid_pts.max(axis=0).astype(int)
    # a window may stick out at most 2 lines past the lines that were found
    xs = range(xmin - 2, max(xmin - 2, xmax - 6) + 1)
    ys = range(ymin - 2, max(ymin - 2, ymax - 6) + 1)
    x_lo, y_lo = xs[0] - 1, ys[0] - 1  # one extra ring of cells to look past the window's edges
    nx, ny = xs[-1] + 9 - x_lo, ys[-1] + 9 - y_lo

    to_px = np.array([[px, 0, -x_lo * px], [0, px, -y_lo * px], [0, 0, 1]], dtype=np.float64)
    M = to_px @ H
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    flat = cv.warpPerspective(lab, M, (nx * px, ny * px)).astype(np.float32)
    inside = cv.warpPerspective(np.ones(img.shape[:2], np.uint8), M, (nx * px, ny * px), flags=cv.INTER_NEAREST)
    m = px // 6  # ignore the cell borders
    cells = flat.reshape(ny, px, nx, px, 3)[:, m:-m, :, m:-m].transpose(0, 2, 1, 3, 4)  # (row, col, y, x, colour)
    visible = inside.reshape(ny, px, nx, px)[:, m:-m, :, m:-m].transpose(0, 2, 1, 3) > 0  # pixels inside the photo

    # Cells past the horizon (seen from a low angle the grid's pattern continues
    # into the background and "behind" the camera) aren't on the table
    to_image = np.linalg.inv(H)
    gx, gy = np.meshgrid(np.arange(nx) + x_lo + 0.5, np.arange(ny) + y_lo + 0.5)
    w = to_image[2, 0] * gx + to_image[2, 1] * gy + to_image[2, 2]
    w_found = to_image[2] @ [grid_pts[:, 0].mean(), grid_pts[:, 1].mean(), 1]
    visible &= (w * w_found > 0)[:, :, None, None]
    share_visible = visible.mean(axis=(2, 3))
    cells_inside = share_visible >= 0.9

    parity = np.add.outer(np.arange(ny) + y_lo, np.arange(nx) + x_lo) % 2
    typical = np.median(cells.reshape(ny, nx, -1, 3), axis=2)  # one colour per cell
    found_x, found_y = set(grid_pts[:, 0].astype(int)), set(grid_pts[:, 1].astype(int))

    best_score, best_xy = -1, None
    for y0 in range(1, ny - 8):
        for x0 in range(1, nx - 8):
            if not cells_inside[y0:y0 + 8, x0:x0 + 8].all():
                continue
            # the whole playing area, corners included, must be in the photo
            gx0, gy0 = x0 + x_lo, y0 + y_lo
            corners = cv.perspectiveTransform(np.float32([[gx0, gy0], [gx0 + 8, gy0], [gx0 + 8, gy0 + 8], [gx0, gy0 + 8]]).reshape(-1, 1, 2), to_image).reshape(-1, 2)
            # (outer lines hidden by the frame are estimated, so allow a quarter square)
            slack = max(2, 0.25 * np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean() / 8)
            if (corners < -slack).any() or (corners[:, 0] > img.shape[1] + slack).any() or (corners[:, 1] > img.shape[0] + slack).any():
                continue
            # Most of the window's 9 + 9 grid lines must be lines that were
            # actually seen, otherwise it's a grid of the wrong size
            if len(found_x & set(range(x0 + x_lo, x0 + x_lo + 9))) < 5 or len(found_y & set(range(y0 + y_lo, y0 + y_lo + 9))) < 5:
                continue
            # The two square colours of this window: the typical colour of
            # each half of the checker pattern
            win, par = cells[y0:y0 + 8, x0:x0 + 8], parity[y0:y0 + 8, x0:x0 + 8]
            colour0 = np.median(typical[y0:y0 + 8, x0:x0 + 8][par == 0], axis=0)
            colour1 = np.median(typical[y0:y0 + 8, x0:x0 + 8][par == 1], axis=0)
            gap = np.linalg.norm(colour0 - colour1)
            if gap < 15:
                continue  # both halves look the same: not a checkerboard
            # For every square of the window and the ring around it: which of
            # the two colours it shows, +1 or -1, or 0 when unsure. Pieces match
            # neither colour, so a square half hidden by a piece still shows its
            # colour; the frame, table and background come out unsure
            r0, c0 = y0 - 1, x0 - 1
            area, seen = cells[r0:y0 + 9, c0:x0 + 9], visible[r0:y0 + 9, c0:x0 + 9]
            count = np.maximum(seen.sum(axis=(2, 3)), 1)
            near0 = ((np.linalg.norm(area - colour0, axis=-1) < gap / 2) & seen).sum(axis=(2, 3)) / count
            near1 = ((np.linalg.norm(area - colour1, axis=-1) < gap / 2) & seen).sum(axis=(2, 3)) / count
            ring = near0 - near1
            ring_seen = share_visible[r0:y0 + 9, c0:x0 + 9] >= 0.15
            light = (near0 if colour0[0] > colour1[0] else near1) * ring_seen  # share of light-square colour
            ring = np.where((np.abs(ring) < 0.15) | ~ring_seen, 0, np.sign(ring))
            shade = ring[y0 - r0:y0 - r0 + 8, x0 - c0:x0 - c0 + 8]
            # If the checker pattern clearly carries on past an edge (the row
            # outside alternates by itself and against the edge row), this
            # window is only part of the board: shifted, or the board is cut
            # off there. A plain frame or table row doesn't alternate.
            outside = [(ring[0, 1:9], shade[0]), (ring[9, 1:9], shade[7]), (ring[1:9, 0], shade[:, 0]), (ring[1:9, 9], shade[:, 7])]
            if any(-(out[:-1] * out[1:]).sum() >= 3 and -(out * edge).sum() >= 3 for out, edge in outside):
                continue
            # Light and darker cells taking turns just past an edge: the board
            # carries on there. A frame is one colour all along (dark wood, or a
            # light band with the coordinates printed on it), real squares
            # alternate even when big pieces near the camera hide half of them
            past = [light[0, 1:9], light[9, 1:9], light[1:9, 0], light[1:9, 9]]
            if any(abs(side[0::2].mean() - side[1::2].mean()) >= 0.3 for side in past):
                continue
            # An edge row that is all one colour is the board's frame (its wood
            # can look like the dark squares), so the window is shifted onto it
            if any(-(edge[:-1] * edge[1:]).sum() <= -3 for _, edge in outside):
                continue
            # On a chessboard neighbours always differ, side by side and above
            # and below: +1 for every pair that differs, -1 for every pair that
            # matches. A grid of the wrong size has neighbours inside the same
            # square, so check both directions and keep the worse one
            across = -(shade[:, :-1] * shade[:, 1:]).mean()
            down = -(shade[:-1, :] * shade[1:, :]).mean()
            score = min(across, down)
            if score > best_score:
                best_score, best_xy = score, (x0 + x_lo, y0 + y_lo)
    return best_xy, best_score


def lines_on_edges(edges, H, xy):
    # Share of the window's inner grid lines (between squares) that lies on
    # edges in the photo. Real square borders show up as edges wherever no
    # piece covers them; a grid built from unrelated lines crosses the squares
    near = cv.dilate(edges, np.ones((3, 3), np.uint8)) > 0
    to_image = np.linalg.inv(H)
    t = np.linspace(0.05, 7.95, 60)
    hits = []
    for k in range(1, 8):
        for pts in (np.column_stack([np.full_like(t, k), t]), np.column_stack([t, np.full_like(t, k)])):
            q = cv.perspectiveTransform((pts + xy).reshape(-1, 1, 2).astype(np.float32), to_image).reshape(-1, 2).round().astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < edges.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < edges.shape[0])
            hits.append(near[q[ok, 1], q[ok, 0]].mean() if ok.any() else 0)
    return float(np.mean(hits))


def order_corners(corners):
    # top-left, top-right, bottom-right, bottom-left (clockwise on screen)
    c = corners - corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(c[:, 1], c[:, 0]))]
    start = np.argmin(corners.sum(axis=1))
    return np.roll(corners, -start, axis=0).astype(np.float32)


def find_board_corners(img, edges, debug=None):
    # Returns the 4 corners of the 8x8 playing area (not the board frame) or None
    min_gap = max(4, 0.015 * min(img.shape[:2]))

    # A grid with half-size squares or one built from diagonals can have just
    # as many crossings as the real one, but it won't look like a checkerboard
    best = (-1, None, None, 0, None, None, None)
    for a, b in family_pairs(edges, min_gap):
        for count, H, grid_pts, img_pts in fit_grid(intersections(a, b), img.shape)[:10]:
            xy, score = pick_8x8(img, H, grid_pts)
            if xy is None or score <= best[0]:
                continue
            on_edges = lines_on_edges(edges, H, xy)
            if on_edges < MIN_LINES_ON_EDGES:
                continue
            best = (score, H, xy, count, (a, b), grid_pts, img_pts)
            if score > 0.5 and on_edges > 0.85:
                break  # clearly the board, no need to look further
        if best[0] > 0.5:
            break
    score, H, xy, count, best_lines, grid_pts, img_pts = best
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
    # Fit the board once more using only the line crossings on it; crossings
    # out in the background pull the fit around
    on_board = (grid_pts[:, 0] >= x0) & (grid_pts[:, 0] <= x0 + 8) & (grid_pts[:, 1] >= y0) & (grid_pts[:, 1] <= y0 + 8)
    if on_board.sum() >= 8:
        refit, _ = cv.findHomography(grid_pts[on_board].astype(np.float32), img_pts[on_board])
        to_image = refit if refit is not None else np.linalg.inv(H)
    else:
        to_image = np.linalg.inv(H)
    board = np.float32([[x0, y0], [x0 + 8, y0], [x0 + 8, y0 + 8], [x0, y0 + 8]])
    corners = cv.perspectiveTransform(board.reshape(-1, 1, 2), to_image).reshape(-1, 2)
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

        if board_corners is not None:
            return split_board(frame, board_corners)
        else:
            print(f"no chessboard detected ({info}), trying again")
            return None


def split_board(frame, board_corners):
        # we divide the board into 64 pieces with angle consideration
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
    # A video file is read much faster than real time, so time it by the video's
    # own clock; otherwise most of each move would be skipped
    from_file = os.path.isfile(source)

    board = BoardTracker(find_board_corners, find_segments)
    tracker = MoveTracker()
    print("Point the camera at the whole board, best before setting up the pieces.")
    print("Keys (in a camera window): q = quit, r = new game, b = find the board again,")
    print("u = take back a wrongly recognised move, m = type the move that was played")
    last_status = None
    last_follow = 0
    corners = None

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Не удалось считать кадр")
            break

        now = camera.get(cv.CAP_PROP_POS_MSEC) / 1000 if from_file else time.time()
        if now - last_follow >= FOLLOW_INTERVAL:
            # Find the board once, then follow it as the camera moves
            was_searching = board.state == 'searching'
            corners = board.update(frame)
            last_follow = now
            if was_searching and board.state == 'locked' and board.empty and corners is not None and tracker.empty is None and tracker.reference is None:
                tracker.set_empty_board(split_board(frame, corners), frame)  # only at the first lock, before the game

            shown = frame.copy()
            if corners is not None:
                cv.polylines(shown, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv.putText(shown, board.status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.imshow('Screenshot', shown)
            if board.status != last_status and not board.status.startswith('Looking'):
                print(board.status)
            last_status = board.status

            if now - LAST_CAPTURE_TIME >= INTERVAL:
                tracker.update(split_board(frame, corners) if corners is not None else None, frame)
                LAST_CAPTURE_TIME = now

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            print("New game: set up the starting position")
        if key == ord('b'):
            board.reset()
            print("Looking for the board again")
        if key == ord('u'):
            tracker.undo()
        if key == ord('m'):
            text = input("Move that was played (e.g. Nf3), Enter to cancel: ").strip()
            if text:
                tracker.play(text)

    camera.release()
    cv.destroyAllWindows()


