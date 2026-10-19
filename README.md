# Chess Board and Piece Detection System


## Features

- **Real-time Processing**: Processes video feed from camera in real-time
- **Visual Feedback**: Shows bounding boxes, labels, and coordinate information

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Full System with Chessboard Detection (`board_finder.py`)

Run:

```bash
python board_finder.py
```

1. Point the camera at the whole board, best while it is still empty, until the
   camera window says "Board locked" (an empty board is found most reliably).
2. Set up the starting position. The game starts once both sides are set up.
3. Play. Each move and the position (FEN) are printed.

After locking, the board is followed as the camera (your head) moves, so it does not
have to be found again in every picture: it keeps working with pieces on it, hands
over it, or part of it out of view. If you look away for a while it finds the board
again when you look back. Instead of the default camera you can pass a camera number, a video
file or a stream URL:

```bash
python board_finder.py 1
python board_finder.py game.mp4
python board_finder.py http://192.168.1.20:8080/video
```

Moves are worked out from what changed on the board and which legal move explains
it, so pieces don't have to be recognised. Works best with the camera looking from
one player's side or from above.


## Output Information

### With Chessboard Detection
- Chessboard bounding box coordinates
- Number of detected squares
- Visual board representation with grid overlay

## Controls

Keys work while a camera window is focused:

- `q`: quit
- `r`: new game (set up the starting position again)
- `b`: find the board again (e.g. after moving the board or the camera a lot)
- `u`: take back a move that was recognised wrongly
- `m`: type the move that was played (in the terminal), e.g. after `u` or when the
  program says it can't work out the move

The board is followed up to 10 times a second; moves are checked every 0.5 seconds,
and a move is read once the board has been still for about 1.5 seconds (no hand over
it).

## Camera Setup

The system uses the default camera (index 0). Make sure:
- Your camera is connected and working
- The chess board is clearly visible
- Good lighting conditions for better detection

## Troubleshooting

1. **Camera not opening**: Check if your camera is connected and not being used by another application
2. **Poor detection**: Improve lighting and ensure the chess board is clearly visible
3. **Chessboard not detected**: 
   - Check if your model was trained to detect chessboards
   - Adjust the class mapping in the script if needed

## File Structure

```
chess/
├── board_finder.py         # Finds the board and splits it into 64 squares, main loop
├── board_tracker.py        # Locks onto the board once, then follows it as the camera moves
├── move_detector.py        # Follows the game move by move
├── test.py                 # Camera test utility
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Python-chess
- Matplotlib
- PyTorch
- TorchVision