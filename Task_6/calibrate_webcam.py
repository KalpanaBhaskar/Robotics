#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_webcam.py
===================

Estimate the MEASURED intrinsic camera matrix and lens distortion coefficients
using pictures of a printed chessboard. This uses cv2.findChessboardCorners()
and cv2.calibrateCamera(). The result is saved to a .npz file that
object_recognition_calibration.py can load with --calib.

Two ways to provide images
--------------------------
  1) Capture live from the webcam:
       python calibrate_webcam.py --capture --rows 6 --cols 9 --square 25
     Hold the chessboard at different distances, angles and positions.
     SPACE saves a frame (only when the board is detected). ESC finishes.

  2) Use images that are already saved in a folder:
       python calibrate_webcam.py --images calib_images --rows 6 --cols 9 --square 25

Board settings
--------------
  --rows, --cols : number of INNER corners (not squares).
                   A board with 7 x 10 squares has 6 x 9 inner corners.
  --square       : size of one square in millimetres. The unit only sets the
                   scale of the translation vectors. It does not change K.

Output
------
  camera_calib.npz containing:
     camera_matrix : 3x3 intrinsic matrix K
     dist_coeffs   : distortion coefficients (k1, k2, p1, p2, k3)
     image_size    : (width, height) used for calibration
     rms           : OpenCV RMS re-projection error (lower is better)

Tip: aim for 15 to 25 good images with the board covering the whole frame,
and an RMS error below about 1.0 pixel.
"""

import argparse
import glob
import os

import cv2
import numpy as np


# Termination criteria for sub-pixel corner refinement
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MIN_IMAGES_REQUIRED = 5          # OpenCV needs several views; 10 to 20 is better


def parse_args():
    parser = argparse.ArgumentParser(description="Chessboard camera calibration with OpenCV.")
    parser.add_argument("--capture", action="store_true",
                        help="Capture chessboard frames from the webcam first.")
    parser.add_argument("--images", default=None,
                        help="Folder with chessboard images (used when --capture is not given).")
    parser.add_argument("--save-folder", default="calib_images",
                        help="Folder where captured frames are saved (default: calib_images).")
    parser.add_argument("--rows", type=int, required=True,
                        help="Number of INNER corners along the board's height.")
    parser.add_argument("--cols", type=int, required=True,
                        help="Number of INNER corners along the board's width.")
    parser.add_argument("--square", type=float, default=25.0,
                        help="Square size in millimetres (default: 25).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default: 0).")
    parser.add_argument("--out", default="camera_calib.npz",
                        help="Output .npz file (default: camera_calib.npz).")
    return parser.parse_args()


def capture_chessboard_frames(folder, pattern_size, camera_index):
    """
    Show the webcam feed. Draw the detected corners as a preview.
    SPACE saves the current frame when the board is detected. ESC finishes.
    Returns the number of frames saved.
    """
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    print("[INFO] SPACE = save frame (board must be detected), ESC = finish capturing.")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, pattern_size, corners, found)
        status = "Board: YES" if found else "Board: no"
        cv2.putText(preview, f"Saved: {saved}   {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Chessboard capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC
            break
        if key == 32 and found:             # SPACE, only when the board is visible
            path = os.path.join(folder, f"calib_{saved:02d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"[INFO] Saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    return saved


def calibrate(image_paths, pattern_size, square_size_mm):
    """
    Run the chessboard calibration.
    Returns (rms, K, dist, image_size, number_of_used_images, mean_reprojection_error).
    """
    cols, rows = pattern_size

    # 3D coordinates of the chessboard corners in the board's own plane (Z = 0).
    # Example for 3 x 2 corners: (0,0,0), (1,0,0), (2,0,0), (0,1,0), ...
    # Multiplying by the square size gives real-world units (mm).
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    object_points = []      # 3D points for each used image (same for all images)
    image_points = []       # 2D corner positions found in each image
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {path}. Skipped.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]                    # OpenCV expects (width, height)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            print(f"[WARN] Chessboard not found in {os.path.basename(path)}. Skipped.")
            continue

        # Refine the corner positions to sub-pixel accuracy for a better calibration
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIXEL_CRITERIA)
        object_points.append(objp)
        image_points.append(corners_refined)
        print(f"[INFO] Chessboard used from {os.path.basename(path)}")

    if len(object_points) < MIN_IMAGES_REQUIRED:
        raise RuntimeError(f"Only {len(object_points)} usable images. "
                           f"At least {MIN_IMAGES_REQUIRED} (preferably 15+) are needed.")

    # Main OpenCV calibration call. It estimates K, the distortion vector,
    # and one rotation and translation per image.
    # CALIB_FIX_K3 keeps the highest-order radial term at zero. With a modest number
    # of images, k3 is poorly constrained and tends to absorb noise.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=cv2.CALIB_FIX_K3)

    # Mean re-projection error: project the 3D points back into each image and
    # measure how far they are from the detected corners.
    total_error = 0.0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        detected_points = np.asarray(image_points[i], dtype=np.float32).reshape(-1, 2)
        projected_points = np.asarray(projected, dtype=np.float32).reshape(-1, 2)
        total_error += cv2.norm(detected_points, projected_points, cv2.NORM_L2) / len(projected_points)
    mean_error = total_error / len(object_points)

    return rms, K, dist, image_size, len(object_points), mean_error


def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)       # OpenCV wants (columns, rows) of inner corners

    if args.capture:
        print("[INFO] Starting webcam capture for calibration...")
        saved = capture_chessboard_frames(args.save_folder, pattern_size, args.camera)
        print(f"[INFO] {saved} frame(s) saved to {args.save_folder}")
        image_folder = args.save_folder
    elif args.images:
        image_folder = args.images
    else:
        raise SystemExit("Give either --capture or --images <folder>.")

    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) +
                         glob.glob(os.path.join(image_folder, "*.png")))
    if not image_paths:
        raise SystemExit(f"No .jpg or .png images found in {image_folder}.")
    print(f"[INFO] {len(image_paths)} image(s) found. Running calibration...")

    rms, K, dist, image_size, used, mean_error = calibrate(
        image_paths, pattern_size, args.square)

    np.savez(args.out,
             camera_matrix=K,
             dist_coeffs=dist,
             image_size=np.array(image_size),
             rms=rms)

    np.set_printoptions(precision=4, suppress=True)
    print("\n" + "=" * 60)
    print("MEASURED CAMERA CALIBRATION RESULT")
    print("=" * 60)
    print(f"Images used               : {used} of {len(image_paths)}")
    print(f"Image size (width, height): {image_size}")
    print("Camera intrinsic matrix K :")
    print(K)
    print("Distortion (k1,k2,p1,p2,k3):", dist.ravel())
    print(f"OpenCV RMS error          : {rms:.4f} px")
    print(f"Mean re-projection error  : {mean_error:.4f} px")
    print(f"Saved to {args.out}")
    print("Use it with: python object_recognition_calibration.py --calib " + args.out + " ...")


if __name__ == "__main__":
    main()
