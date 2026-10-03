# Follows a game move by move. It doesn't recognise pieces: it compares the
# board before and after a move, finds where the picture changed, and asks
# python-chess which legal move explains that change best.
# Start it with the pieces in the starting position.

import cv2 as cv
import numpy as np
import chess

STEADY_FRAMES = 3     # the board must look the same in this many captures in a row
CHANGED_PIXEL = 30    # a pixel counts as changed when it differs by more than this (0-255)
STEADY_AREA = 0.3     # less changed area than this (in squares) between two captures = steady
MIN_PIECE_AREA = 0.1  # every square of a move must show at least this much changed area (in squares)
MAX_MOVE_AREA = 8     # more changed area than this (in squares) = something covers the board
PIECE_HEIGHT = 1.2    # piece height in squares (tall pieces), used to work out where pieces show up
SQUARE_PX = 48        # size of a square in the top-down picture
MARGIN = 2            # squares of table shown around the board (pieces lean out over the edge)
SCALE = 4             # change maps are worked out at 1/SCALE resolution


def top_down(frame, squares_grid):
    # Top-down picture of the board with MARGIN squares around it, so pieces on
    # the outer ranks that lean out over the board's edge are still in view.
    # Also returns the homography from this picture back to the camera frame, and
    # a 1/SCALE mask of the part the camera actually saw (the margin can reach
    # past the edge of the frame).
    corners = np.float32([squares_grid[0]['coordinates_original'][0], squares_grid[7]['coordinates_original'][1],
                          squares_grid[63]['coordinates_original'][2], squares_grid[56]['coordinates_original'][3]])
    m, n = MARGIN * SQUARE_PX, (8 + 2 * MARGIN) * SQUARE_PX
    board = np.float32([[m, m], [m + 8 * SQUARE_PX, m], [m + 8 * SQUARE_PX, m + 8 * SQUARE_PX], [m, m + 8 * SQUARE_PX]])
    H = cv.getPerspectiveTransform(corners, board)
    seen = cv.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), H, (n, n))
    seen = cv.erode(cv.resize(seen, (n // SCALE, n // SCALE), interpolation=cv.INTER_AREA), np.ones((3, 3), np.uint8)) == 255
    return cv.warpPerspective(frame, H, (n, n)), np.linalg.inv(H), seen


def square_center(row, col):
    return np.array([(MARGIN + col + 0.5) * SQUARE_PX, (MARGIN + row + 0.5) * SQUARE_PX])


def piece_leans(to_camera):
    # Seen at an angle, a standing piece doesn't stay on its square in the
    # top-down picture: it leans away from the camera, the lower the camera the
    # longer. For every square, work out the vector (top-down pixels) from the
    # square's centre to where the top of a piece standing there shows up.
    # The camera angle comes from how squashed the squares look.
    leans = np.zeros((8, 8, 2))
    for r in range(8):
        for c in range(8):
            center = square_center(r, c)
            pts = np.float32([center, center + [1, 0], center + [0, 1]]).reshape(-1, 1, 2)
            p = cv.perspectiveTransform(pts, to_camera).reshape(-1, 2)
            J = np.column_stack([p[1] - p[0], p[2] - p[0]])  # camera pixels per top-down pixel
            up = np.linalg.solve(J, [0, -1.0])               # top-down step that moves one camera pixel up
            away = up / np.linalg.norm(up)                   # direction a standing piece leans
            side = np.array([-away[1], away[0]])
            # squares look squashed along "away" by sin(camera elevation)
            sin_e = np.clip((1 / np.linalg.norm(up)) / np.linalg.norm(J @ side), 0.2, 1.0)
            lean = PIECE_HEIGHT * SQUARE_PX * np.sqrt(1 - sin_e ** 2) / sin_e
            leans[r, c] = away * min(lean, 2 * SQUARE_PX)
    return leans


def rotate(img, turns):
    return np.ascontiguousarray(np.rot90(img, turns))


def rotate_leans(leans, turns):
    # turn the 8x8 grid of vectors the same way as rotate() turns the picture
    for _ in range(turns % 4):
        leans = np.rot90(leans)
        leans = np.stack([leans[..., 1], -leans[..., 0]], axis=-1)
    return leans


def piece_bases():
    # For every square, a mask of the spot around its centre where the bottom of
    # a piece stands. Shape (8, 8, h, w), 1/SCALE resolution
    n = (8 + 2 * MARGIN) * SQUARE_PX // SCALE
    bases = np.zeros((8, 8, n, n), np.uint8)
    for r in range(8):
        for c in range(8):
            cv.circle(bases[r, c], tuple((square_center(r, c) / SCALE).astype(int)), int(0.3 * SQUARE_PX / SCALE), 1, -1)
    return bases.astype(bool)


BASES = piece_bases()


def piece_columns(leans):
    # For every square, a mask of where a piece standing there can show up:
    # its base plus the strip it leans into. Shape (8, 8, h, w), 1/SCALE resolution
    columns = BASES.astype(np.uint8)
    for r in range(8):
        for c in range(8):
            base = square_center(r, c) / SCALE
            top = base + leans[r, c] / SCALE
            cv.line(columns[r, c], tuple(base.astype(int)), tuple(top.astype(int)), 1, max(1, int(0.6 * SQUARE_PX / SCALE)))
    return columns.astype(bool)


def change_map(before, after):
    # True where the picture changed (1/SCALE resolution)
    diff = cv.absdiff(cv.GaussianBlur(before, (7, 7), 0), cv.GaussianBlur(after, (7, 7), 0)).max(axis=2)
    h, w = diff.shape
    diff = cv.resize(diff, (w // SCALE, h // SCALE), interpolation=cv.INTER_AREA).astype(np.float32)
    changed = (diff > CHANGED_PIXEL + np.median(diff)).astype(np.uint8)  # the median removes overall lighting shifts
    # Drop specks (tips of tall pieces shift a little when the camera moves) and
    # the outer edge, where lining the pictures up leaves small seams
    changed = cv.morphologyEx(changed, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed[:2], changed[-2:], changed[:, :2], changed[:, -2:] = 0, 0, 0, 0
    return changed.astype(bool)


def find_orientation(img):
    # Quarter turns that put white at the bottom, worked out from the starting
    # position: the rows with pieces are busy, the empty middle is plain,
    # and white's side is the brighter one
    s, edge = SQUARE_PX, MARGIN * SQUARE_PX
    gray = cv.cvtColor(img[edge:edge + 8 * s, edge:edge + 8 * s], cv.COLOR_BGR2GRAY).astype(np.float32)
    m = s // 5
    cells = gray.reshape(8, s, 8, s)[:, m:s - m, :, m:s - m]
    busy = cells.std(axis=(1, 3))
    bright = cells.mean(axis=(1, 3))

    def pieces_in_rows(turns):
        b = np.rot90(busy, turns)
        return b[[0, 1, 6, 7]].mean() - b[2:6].mean()

    turns = 0 if pieces_in_rows(0) >= pieces_in_rows(1) else 1
    rows = np.rot90(bright, turns)
    if rows[6:].mean() < rows[:2].mean():
        turns += 2
    return turns


def best_rotation(reference, img):
    # the camera may turn (head movement), so the detected corners can come in
    # a different order; pick the quarter turn that best matches the last known board
    diffs = [cv.absdiff(reference, rotate(img, k)).mean() for k in range(4)]
    return int(np.argmin(diffs))


def align(reference, img):
    # The detected grid wobbles by a few pixels between captures (most of all at
    # the far edge, where pieces hide the last grid line). Line the picture up
    # with the reference so that wobble doesn't look like pieces moving
    ref_gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp = cv.findTransformECC(ref_gray, img_gray, warp, cv.MOTION_HOMOGRAPHY, criteria, None, 5)
    except cv.error:
        return img  # didn't converge, compare as it is
    h, w = reference.shape[:2]
    return cv.warpPerspective(img, warp, (w, h), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)


def square_position(square):
    # python-chess square -> picture row/column (white at the bottom)
    return 7 - chess.square_rank(square), chess.square_file(square)


def squares_touched(board, move):
    # every square that looks different after this move
    touched = {move.from_square, move.to_square}
    rank = chess.square_rank(move.from_square)
    if board.is_kingside_castling(move):
        touched |= {chess.square(7, rank), chess.square(5, rank)}
    elif board.is_queenside_castling(move):
        touched |= {chess.square(0, rank), chess.square(3, rank)}
    elif board.is_en_passant(move):
        touched.add(chess.square(chess.square_file(move.to_square), rank))
    return touched


def candidates(board, depth):
    # (moves, squares they touch) for every legal move, or with depth=2 every
    # legal move plus every reply to it
    for first in board.legal_moves:
        if first.promotion not in (None, chess.QUEEN):
            continue  # the camera can't tell what a pawn promoted to, assume a queen
        touched = squares_touched(board, first)
        if depth == 1:
            yield [first], touched
            continue
        board.push(first)
        replies = [(m, squares_touched(board, m)) for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
        board.pop()
        for reply, reply_touched in replies:
            yield [first, reply], touched | reply_touched


def guess_moves(board, changed, columns, square_area, depth=1):
    # Score every legal move (or move + reply) by how well the columns of its
    # squares cover the changed pixels: changes inside them count for it,
    # changes outside count against it, unchanged area it claims a little
    # against it. On top of that every square involved votes with its base (the
    # spot the piece stands on): a piece arriving or leaving nearly always
    # changes a good part of it, so a base that stayed the same is strong
    # evidence against the move. Every square involved must show some change.
    # Returns the list of moves or None
    total = changed.sum()
    change_near = {s: (changed & columns[square_position(s)]).sum() for s in chess.SQUARES}
    base_changed = {s: (changed & BASES[square_position(s)]).mean() / BASES[square_position(s)].mean() for s in chess.SQUARES}
    scored = []
    for moves, touched in candidates(board, depth):
        if min(change_near[s] for s in touched) < MIN_PIECE_AREA * square_area:
            continue
        claimed = np.logical_or.reduce([columns[square_position(s)] for s in touched])
        explained = (changed & claimed).sum()
        base_votes = sum(base_changed[s] - 0.3 for s in touched) * 0.5 * square_area
        score = explained - (total - explained) - 0.05 * (claimed & ~changed).sum() + base_votes
        scored.append((score, explained, moves))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    score, explained, moves = scored[0]
    if explained < 0.75 * total:
        return None  # a good part of the change isn't explained (another move, a hand...)
    if len(scored) > 1 and score - scored[1][0] < 0.15 * depth * square_area:
        return None  # two guesses fit about equally well, wait for a clearer picture
    return moves


def changed_names(changed, columns, square_area):
    names = []
    for square in chess.SQUARES:
        if (changed & columns[square_position(square)]).sum() >= MIN_PIECE_AREA * square_area:
            names.append(chess.square_name(square))
    return ', '.join(names)


class MoveTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.reference = None  # last picture of a known position, white at the bottom
        self.reference_seen = None
        self.last = None       # previous capture, to tell when the board is steady
        self.last_seen = None
        self.steady = 0
        self.warned = None     # don't repeat the same warning every capture
        self.last_touched = None  # squares of the last recognised move
        self.history = []      # (number of moves, reference before them) for undo
        self.paused = False    # after an undo, wait for the user to type the right move

    def warn(self, text):
        if text != self.warned:
            print(text)
            self.warned = text

    def push(self, moves, reference, reference_seen):
        self.history.append((len(moves), self.reference, self.reference_seen))
        self.last_touched = set()
        for move in moves:
            number = f"{self.board.fullmove_number}." if self.board.turn == chess.WHITE else f"{self.board.fullmove_number}..."
            print(f"Move {number} {self.board.san(move)}")
            self.last_touched |= squares_touched(self.board, move)
            self.board.push(move)
        print(f"FEN: {self.board.fen()}")
        self.reference, self.reference_seen = reference, reference_seen
        self.warned = None

    def undo(self):
        # Take back the last recognised move(s), e.g. when the tracker got one wrong
        if not self.history:
            print("Nothing to take back")
            return
        count, self.reference, self.reference_seen = self.history.pop()
        taken = [self.board.pop() for _ in range(count)]
        self.last_touched = None
        self.paused = True
        print(f"Took back {', '.join(m.uci() for m in reversed(taken))}. Press m to type the move that was played")

    def play(self, text):
        # The user typed the move that was played (one the tracker missed or got
        # wrong). The board as the camera sees it now becomes the new reference
        try:
            move = self.board.parse_san(text)
        except ValueError:
            print(f"'{text}' isn't a legal move here (position: {self.board.fen()})")
            return
        self.paused = False
        if self.last is not None:
            self.push([move], self.last, self.last_seen)
        else:
            self.push([move], self.reference, self.reference_seen)

    def update(self, squares_grid, frame):
        # Call with every result of detect_chessboard and the camera frame it came
        # from. Returns the list of moves recognised (already played on
        # self.board; usually one, two if both players moved in between), or None
        if squares_grid is None:
            self.steady, self.last = 0, None
            return None
        if self.paused:
            return None

        img, to_camera, seen = top_down(frame, squares_grid)
        leans = piece_leans(to_camera)
        anchor = self.reference if self.reference is not None else self.last
        if anchor is not None:
            turns = best_rotation(anchor, img)
            img, seen, leans = rotate(img, turns), rotate(seen, turns), rotate_leans(leans, turns)
        columns = piece_columns(leans)
        look = columns.any(axis=(0, 1)) & seen  # only where a piece can show up and the camera can see
        square_area = (SQUARE_PX / SCALE) ** 2

        # Only look at the board once it has stopped changing (no hand moving over it)
        moving = None if self.last is None else change_map(self.last, align(self.last, img)) & look & self.last_seen
        if moving is not None and moving.sum() < STEADY_AREA * square_area:
            self.steady += 1
        else:
            self.steady = 1
        self.last, self.last_seen = img, seen
        if self.steady < STEADY_FRAMES:
            return None

        if self.reference is None:
            turns = find_orientation(img)
            self.reference, self.reference_seen = rotate(img, turns), rotate(seen, turns)
            self.last, self.last_seen = self.reference, self.reference_seen
            print("Starting position recorded, waiting for white's first move")
            return None

        img = align(self.reference, img)
        changed = change_map(self.reference, img) & look & self.reference_seen
        shown = img.copy()
        shown[cv.resize(changed.astype(np.uint8), img.shape[1::-1], interpolation=cv.INTER_NEAREST) > 0] = (0, 0, 255)
        cv.imshow('Move tracker (white at the bottom)', shown)

        area = changed.sum() / square_area
        if area < MIN_PIECE_AREA:
            return None
        if area > MAX_MOVE_AREA:
            self.warn("Something is covering the board, waiting")
            return None

        moves = guess_moves(self.board, changed, columns, square_area)
        if moves is None:
            if self.last_touched:
                # Changes only on the squares of the move just recognised: the
                # player paused mid-capture and has now put the piece down. The
                # opponent's move can't look like this (it has to start on
                # another square), so just take the new picture
                last = np.logical_or.reduce([columns[square_position(s)] for s in self.last_touched])
                if (changed & ~last).sum() < MIN_PIECE_AREA * square_area:
                    self.reference, self.reference_seen = img, seen
                    return None
            # Maybe both players moved before the board was still in between
            moves = guess_moves(self.board, changed, columns, square_area, depth=2)
        if moves is None:
            self.warn(f"Can't work out the move yet (changes near: {changed_names(changed, columns, square_area)})")
            return None

        self.push(moves, img, seen)
        return moves
