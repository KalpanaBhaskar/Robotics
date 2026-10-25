#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sample_materials.py
========================

Generates every image needed to run the assignment, without a real camera.

Output (default folder: sample_materials/)
    reference.jpg                      clean reference image of the "product box"
    query_similar_viewpoint.jpg        same box, almost the same angle
    query_different_viewpoint.jpg      same box, strong change of angle and distance
    query_different_object.jpg         a different box (should NOT be recognized)
    chessboard/calib_00.jpg ...        20 chessboard views for calibrate_webcam.py
    camera_truth.txt                   the "true" camera used to make the images

How the camera is simulated
---------------------------
The query and chessboard images are taken by a simulated camera with a KNOWN
intrinsic matrix and KNOWN lens distortion:

    K    = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    dist = (k1, k2, p1, p2, k3) = (-0.25, 0.08, 0.0, 0.0, 0.0)

The reference image is a clean digital image (as if it came from a scanner or
a design file), so it has no distortion. Because the lens distortion is known,
calibrate_webcam.py should recover values close to these, and cv2.undistort()
should visibly straighten the query images once a measured calibration is used.

Usage
-----
    python make_sample_materials.py                 # writes into sample_materials/
    python make_sample_materials.py --out my_folder
"""

import argparse
import os

import cv2
import numpy as np


IMG_W, IMG_H = 640, 480

# The simulated "true" camera (used only to make the images)
TRUE_K = np.array([[500.0, 0.0, 320.0],
                   [0.0, 500.0, 240.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
TRUE_DIST = np.array([-0.25, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)

NO_DIST = np.zeros(5, dtype=np.float64)   # ideal pinhole projection

BOARD_COLS, BOARD_ROWS = 9, 6    # inner corners (as used by calibrate_webcam.py)
SQUARE_MM = 25.0


def make_product_box_face(seed, title, subtitle, base_colors):
    """
    Draw a synthetic product-box face with plenty of texture so that SIFT
    can find many distinctive landmarks: colour blocks, shapes, text, a logo
    and a barcode-like pattern.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)

    # Vertical colour gradient background
    top, bottom = base_colors
    for y in range(IMG_H):
        t = y / (IMG_H - 1)
        img[y, :] = np.array(top) * (1 - t) + np.array(bottom) * t

    # Random coloured shapes give many corners and blobs for SIFT
    for _ in range(60):
        color = tuple(int(c) for c in rng.integers(30, 255, 3))
        x1, y1 = (int(v) for v in rng.integers(0, [IMG_W, IMG_H]))
        if rng.random() < 0.5:
            x2, y2 = x1 + int(rng.integers(20, 120)), y1 + int(rng.integers(20, 90))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.circle(img, (x1, y1), int(rng.integers(8, 45)), color, -1)

    # Big title with a dark outline, so the title itself has strong edges
    cv2.putText(img, title, (30, 150), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 12, cv2.LINE_AA)
    cv2.putText(img, title, (30, 150), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, subtitle, (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Small text lines (good for landmarks)
    for i in range(4):
        cv2.putText(img, f"Item {rng.integers(100, 999)}  net wt {rng.integers(50, 500)} g",
                    (30, 280 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # Logo: circle with a letter
    logo_colour = tuple(int(c) for c in rng.integers(0, 255, 3))
    cv2.circle(img, (540, 90), 55, logo_colour, -1)
    cv2.circle(img, (540, 90), 55, (255, 255, 255), 4)
    cv2.putText(img, title[0], (514, 110), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)

    # Barcode-like stripes (strong vertical edges)
    x = 380
    while x < 600:
        w = int(rng.integers(2, 7))
        cv2.rectangle(img, (x, 390), (x + w, 450), (0, 0, 0), -1)
        x += w + int(rng.integers(2, 7))
    return img


def make_other_object(seed):
    """
    A DIFFERENT product design. It deliberately shares none of the reference's
    layout elements: no barcode, no "Item" text lines, no logo circle. It uses a
    dark background, diagonal stripes, a script font and star shapes, so its
    landmarks should not match the reference landmarks.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    img[:] = (25, 25, 30)                                  # dark charcoal background
    # Diagonal stripes
    for x in range(-IMG_H, IMG_W, 46):
        cv2.line(img, (x, 0), (x + IMG_H, IMG_H), (60, 60, 70), 18)
    # Star / triangle shapes in random bright colours
    for _ in range(35):
        cx, cy = (int(v) for v in rng.integers(0, [IMG_W, IMG_H]))
        r = int(rng.integers(12, 40))
        pts = np.array([[cx + r * np.cos(a), cy + r * np.sin(a)]
                        for a in np.linspace(0, 2 * np.pi, 7)[:-1] + rng.uniform(0, 1)], np.int32)
        cv2.fillPoly(img, [pts], tuple(int(c) for c in rng.integers(80, 255, 3)))
    # Title in a script font with a light outline
    cv2.putText(img, "Fresh Leaf", (60, 200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.2, (0, 0, 0), 14, cv2.LINE_AA)
    cv2.putText(img, "Fresh Leaf", (60, 200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.2, (120, 255, 170), 4, cv2.LINE_AA)
    # Speckle texture so the surface is not flat
    noise = rng.integers(0, 40, (IMG_H, IMG_W, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    return img


def apply_lens_distortion(ideal_img, K, dist, fill=(255, 255, 255)):
    """
    Simulate the lens: move every ideal pixel to the distorted position that the
    camera would produce. For each pixel of the distorted output, cv2.undistortPoints
    gives the corresponding ideal location, and we sample the ideal image there.
    """
    h, w = ideal_img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    distorted_grid = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    ideal_pts = cv2.undistortPoints(distorted_grid, K, dist, P=K).reshape(h, w, 2)
    map_x = ideal_pts[..., 0].astype(np.float32)
    map_y = ideal_pts[..., 1].astype(np.float32)
    return cv2.remap(ideal_img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def viewpoint_warp(img, dst_corners):
    """Simulate a change of viewpoint: warp the flat object with a perspective transform."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, np.float32(dst_corners))
    return cv2.warpPerspective(img, H, (w, h), borderValue=(255, 255, 255))


def render_chessboard_view(rvec, tvec):
    """
    Draw the chessboard as an IDEAL (undistorted) pinhole image. The lens distortion
    is added afterwards, once, by apply_lens_distortion(). Distorting here as well
    would apply the lens effect twice and the calibration would recover wrong values.
    """
    canvas = np.full((IMG_H, IMG_W, 3), 110, np.uint8)     # grey table background
    n_cols, n_rows = BOARD_COLS + 1, BOARD_ROWS + 1         # squares = inner corners + 1

    # White board outline first (so the outer border is visible, as on a real board)
    outline = np.float32([[0, 0, 0], [n_cols * SQUARE_MM, 0, 0],
                          [n_cols * SQUARE_MM, n_rows * SQUARE_MM, 0],
                          [0, n_rows * SQUARE_MM, 0]])
    poly, _ = cv2.projectPoints(outline, rvec, tvec, TRUE_K, NO_DIST)
    cv2.fillConvexPoly(canvas, poly.reshape(-1, 2).astype(np.int32), (255, 255, 255))

    # Black squares in a checker pattern
    for r in range(n_rows):
        for c in range(n_cols):
            if (r + c) % 2 == 0:
                x, y = c * SQUARE_MM, r * SQUARE_MM
                sq = np.float32([[x, y, 0], [x + SQUARE_MM, y, 0],
                                 [x + SQUARE_MM, y + SQUARE_MM, 0], [x, y + SQUARE_MM, 0]])
                poly, _ = cv2.projectPoints(sq, rvec, tvec, TRUE_K, NO_DIST)
                cv2.fillConvexPoly(canvas, poly.reshape(-1, 2).astype(np.int32), (0, 0, 0))
    return canvas


def board_fully_visible(rvec, tvec, margin=12):
    """True if the whole chessboard (its outer outline) lies inside the image."""
    n_cols, n_rows = BOARD_COLS + 1, BOARD_ROWS + 1
    outline = np.float32([[0, 0, 0], [n_cols * SQUARE_MM, 0, 0],
                          [n_cols * SQUARE_MM, n_rows * SQUARE_MM, 0],
                          [0, n_rows * SQUARE_MM, 0]])
    pts, _ = cv2.projectPoints(outline, rvec, tvec, TRUE_K, TRUE_DIST)   # real camera
    pts = pts.reshape(-1, 2)
    return bool(np.all(pts[:, 0] > margin) and np.all(pts[:, 0] < IMG_W - margin) and
                np.all(pts[:, 1] > margin) and np.all(pts[:, 1] < IMG_H - margin))


def generate_chessboard_set(folder, count, rng):
    """
    Create `count` chessboard views that together cover the WHOLE frame, including
    the corners, where lens distortion is strongest. Views where the full board would
    leave the frame are rejected, because the detector needs the whole board.
    """
    os.makedirs(folder, exist_ok=True)
    board_cx = (BOARD_COLS + 1) * SQUARE_MM / 2.0
    board_cy = (BOARD_ROWS + 1) * SQUARE_MM / 2.0
    saved, tries = 0, 0
    while saved < count and tries < 5000:
        tries += 1
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        z = rng.uniform(260, 420)
        # Wide range of positions so the board can appear near every corner of the frame
        tvec = np.array([-board_cx + rng.uniform(-260, 260),
                         -board_cy + rng.uniform(-200, 200),
                         z])
        rv, tv = rvec.reshape(3, 1), tvec.reshape(3, 1)
        if not board_fully_visible(rv, tv):
            continue
        img = render_chessboard_view(rv, tv)
        img = apply_lens_distortion(img, TRUE_K, TRUE_DIST, fill=(110, 110, 110))
        cv2.imwrite(os.path.join(folder, f"calib_{saved:02d}.jpg"), img)
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic images for the assignment.")
    parser.add_argument("--out", default="sample_materials", help="Output folder")
    args = parser.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2024)

    # 1) Reference object: clean image of the product box (no lens distortion)
    reference = make_product_box_face(
        seed=11, title="CRUNCHY", subtitle="Oat & Honey Granola",
        base_colors=((40, 90, 200), (20, 160, 230)))
    cv2.imwrite(os.path.join(out, "reference.jpg"), reference)

    # 2) Query: same box, almost the same angle (small shift, small tilt)
    similar_ideal = viewpoint_warp(reference, [[30, 20], [IMG_W - 25, 35], [IMG_W - 15, IMG_H - 25], [25, IMG_H - 30]])
    similar = apply_lens_distortion(similar_ideal, TRUE_K, TRUE_DIST)
    cv2.imwrite(os.path.join(out, "query_similar_viewpoint.jpg"), similar)

    # 3) Query: same box, from a much different angle and distance
    different_ideal = viewpoint_warp(reference, [[90, 70], [IMG_W - 120, 20], [IMG_W - 60, IMG_H - 60], [60, IMG_H - 40]])
    different = apply_lens_distortion(different_ideal, TRUE_K, TRUE_DIST)
    cv2.imwrite(os.path.join(out, "query_different_viewpoint.jpg"), different)

    # 4) A different object (another box design)
    other_ideal = make_other_object(seed=77)
    other = apply_lens_distortion(other_ideal, TRUE_K, TRUE_DIST)
    cv2.imwrite(os.path.join(out, "query_different_object.jpg"), other)

    # 5) Chessboard calibration set (20 views, with lens distortion)
    n = generate_chessboard_set(os.path.join(out, "chessboard"), 20, rng)

    with open(os.path.join(out, "camera_truth.txt"), "w", encoding="utf-8") as f:
        f.write("Simulated camera used to generate the query and chessboard images.\n")
        f.write("This is the answer key. Your measured calibration should be close to it.\n\n")
        f.write("Camera matrix K:\n" + np.array2string(TRUE_K, precision=2) + "\n\n")
        f.write("Distortion (k1, k2, p1, p2, k3): " + str(TRUE_DIST.tolist()) + "\n")
        f.write(f"Image size: {IMG_W} x {IMG_H}\n")
        f.write(f"Chessboard: {BOARD_COLS} x {BOARD_ROWS} inner corners, {SQUARE_MM} mm squares\n")

    print(f"Reference, 3 query images and {n} chessboard images written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
