import cv2
import mediapipe as mp
import pygame
import math
import os
import sys

# --- Config ---
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8
FPS = 60

# Colors
BOARD_LIGHT = (235, 235, 208)
BOARD_DARK = (119, 149, 86)
LANDMARK_DOT = (255, 255, 255) # White Dots
LANDMARK_LINE = (200, 0, 0)     # Red Lines

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Virtual Hand Chess")
clock = pygame.time.Clock()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

# --- Piece & Image Logic ---
IMAGES = {}

def load_and_clean_images():
    """Loads images and removes white background automatically"""
    piece_names = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for name in piece_names:
        path = f"assets/{name}.png"
        if os.path.exists(path):
            img = pygame.image.load(path).convert() # Use .convert() for colorkey
            # Set white (255, 255, 255) as transparent
            img.set_colorkey((255, 255, 255))
            IMAGES[name] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        else:
            print(f"Warning: {path} not found!")

def create_board():
    pieces = []
    back_rank = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i in range(8):
        # White (w) and Black (b)
        pieces.append({"id": f"w{back_rank[i]}", "pos": [i*SQUARE_SIZE+50, 7*SQUARE_SIZE+50], "dragging": False})
        pieces.append({"id": f"wP", "pos": [i*SQUARE_SIZE+50, 6*SQUARE_SIZE+50], "dragging": False})
        pieces.append({"id": f"b{back_rank[i]}", "pos": [i*SQUARE_SIZE+50, 0*SQUARE_SIZE+50], "dragging": False})
        pieces.append({"id": f"bP", "pos": [i*SQUARE_SIZE+50, 1*SQUARE_SIZE+50], "dragging": False})
    return pieces

load_and_clean_images()
all_pieces = create_board()

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. Draw Chessboard
    for r in range(8):
        for c in range(8):
            color = BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK
            pygame.draw.rect(screen, color, (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # 2. Hand Processing & Custom Visualization
    cursor = None
    is_pinching = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            ix, iy = int(lm[8].x * WIDTH), int(lm[8].y * HEIGHT)
            tx, ty = int(lm[4].x * WIDTH), int(lm[4].y * HEIGHT)
            cursor = (ix, iy)
            is_pinching = math.hypot(ix - tx, iy - ty) < 45

            # Draw Red Lines (Hand Skeleton)
            for conn in mp_hands.HAND_CONNECTIONS:
                start = (int(lm[conn[0]].x * WIDTH), int(lm[conn[0]].y * HEIGHT))
                end = (int(lm[conn[1]].x * WIDTH), int(lm[conn[1]].y * HEIGHT))
                pygame.draw.line(screen, LANDMARK_LINE, start, end, 2)

            # Draw White Dots (Joints)
            for l in lm:
                pygame.draw.circle(screen, LANDMARK_DOT, (int(l.x * WIDTH), int(l.y * HEIGHT)), 5)

    # 3. Pinch-to-Move Logic
    for p in all_pieces:
        if cursor:
            dist = math.hypot(cursor[0] - p["pos"][0], cursor[1] - p["pos"][1])
            if is_pinching and (dist < 50 or p["dragging"]):
                # Allow only one piece to be dragged
                if not any(other["dragging"] for other in all_pieces if other != p):
                    p["pos"] = list(cursor)
                    p["dragging"] = True
            elif p["dragging"]:
                # Snap to Center
                col = max(0, min(7, p["pos"][0] // SQUARE_SIZE))
                row = max(0, min(7, p["pos"][1] // SQUARE_SIZE))
                p["pos"] = [col * SQUARE_SIZE + 50, row * SQUARE_SIZE + 50]
                p["dragging"] = False

    # 4. Draw Pieces
    for p in all_pieces:
        if p["id"] in IMAGES:
            img = IMAGES[p["id"]]
            screen.blit(img, (p["pos"][0] - SQUARE_SIZE//2, p["pos"][1] - SQUARE_SIZE//2))

    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release(); pygame.quit(); sys.exit()
    clock.tick(FPS)