#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
object_recognition_calibration.py
=================================

Object recognition using SIFT landmarks, Brute-Force KNN matching with
Lowe's ratio test, and lens-distortion correction with the camera
intrinsic (calibration) matrix. Built with OpenCV and NumPy.

Pipeline
--------
 1. Load the reference image and the query image, convert both to grayscale.
 2. Build an APPROXIMATE camera matrix K from the image size, or load a
    MEASURED one from a .npz file made by calibrate_webcam.py.
 3. Undistort the query image with cv2.undistort().
 4. Detect SIFT keypoints (landmarks) and descriptors in the reference and
    in the query (both before and after undistortion, for comparison).
 5. Match descriptors with Brute-Force L2 KNN (k=2) and keep matches that pass
    Lowe's ratio test.
 6. Decide OBJECT RECOGNIZED / OBJECT NOT RECOGNIZED from the good-match count.
 7. Save the undistorted query, keypoint images, the match image and a
    results report, and append one row to observations.csv.
 8. Optional extension (--homography): estimate the object's location with
    cv2.findHomography() + RANSAC and draw its outline on the query image.

Run
---
  # Use two saved photos
  python object_recognition_calibration.py --reference reference.jpg --query query.jpg

  # Capture the query live from the webcam (SPACE = capture, ESC = cancel)
  python object_recognition_calibration.py --reference reference.jpg

  # Use a measured calibration file (made by calibrate_webcam.py)
  python object_recognition_calibration.py --reference reference.jpg --query query.jpg --calib camera_calib.npz

  # Add the homography / object-location extension
  python object_recognition_calibration.py --reference reference.jpg --query query.jpg --homography

  # Label the run for the observation table and do not open windows
  python object_recognition_calibration.py --reference reference.jpg --query q2.jpg --label "different viewpoint" --no-display
"""

import argparse
import csv
import datetime
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Configuration constants. Tune them if needed, and record any change you make.
# ----------------------------------------------------------------------------
LOWE_RATIO = 0.75                # Lowe's ratio test threshold (0.7 - 0.8 is typical)
MIN_GOOD_MATCHES = 50            # At least this many good matches -> OBJECT RECOGNIZED.
                                 # Tuned on the sample set: same object 400+, different object ~40.
                                 # Tune it again for your own object and camera.
MIN_HOMOGRAPHY_INLIERS = 20      # RANSAC inliers needed to accept the object location
RANSAC_REPROJ_THRESHOLD = 5.0    # Pixel tolerance used by RANSAC
DEFAULT_OUTPUT_DIR = "outputs"   # All saved images and reports go here


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def print_section(title):
    """Print a simple heading so the console output is easy to read."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ensure_output_dir(path):
    """Create the output folder if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def build_approximate_camera_matrix(image_width, image_height, focal_length_px=None):
    """
    Build an APPROXIMATE camera intrinsic matrix K.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    fx, fy : focal length in pixels. If it is not given, we use the image
             width, which corresponds to a horizontal field of view of about
             53 degrees (f = w / (2 * tan(FOV / 2))). This is a rough guess
             that works reasonably for many webcams.
    cx, cy : principal point. We assume it is at the centre of the image.

    The distortion vector is set to zeros, which means "assume no lens
    distortion". This is ONLY an estimate. It must NOT be reported as a
    measured calibration result.
    """
    if focal_length_px is None:
        focal_length_px = float(image_width)

    fx = fy = float(focal_length_px)
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # Distortion coefficients in OpenCV order: (k1, k2, p1, p2, k3)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def load_calibration(npz_path):
    """
    Load a MEASURED calibration saved by calibrate_webcam.py.
    Returns K, dist and the image size used during calibration (width, height).
    """
    data = np.load(npz_path)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    calib_size = tuple(int(v) for v in data["image_size"]) if "image_size" in data else None
    return K, dist, calib_size


def rescale_camera_matrix(K, calib_size, image_size):
    """
    A camera matrix is only valid for the resolution it was calibrated at.
    If the query image has a different resolution, scale fx, cx (by width ratio)
    and fy, cy (by height ratio).
    """
    if calib_size is None:
        return K
    calib_w, calib_h = calib_size
    img_w, img_h = image_size
    if (calib_w, calib_h) == (img_w, img_h):
        return K

    sx = img_w / calib_w
    sy = img_h / calib_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    print(f"[INFO] Calibration was done at {calib_w}x{calib_h}; "
          f"query is {img_w}x{img_h}. K was rescaled to match.")
    return K_scaled


def capture_query_from_webcam(camera_index=0):
    """
    Show the live webcam feed.
      SPACE -> capture the current frame and return it
      ESC   -> cancel and return None
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}. "
                           "Try another index with --camera.")

    print("[INFO] Webcam is open. Press SPACE to capture the query image, ESC to cancel.")
    captured = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read a frame from the webcam.")
            break

        cv2.imshow("Webcam - SPACE to capture, ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC key
            break
        if key == 32:            # SPACE key
            captured = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def load_color_and_gray(path):
    """Read an image with OpenCV and return (color image, grayscale image)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read the image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_landmarks(gray, sift):
    """
    Detect SIFT keypoints (the 'landmarks') and compute their 128-value
    descriptors. If no keypoints are found, return an empty descriptor array
    so the rest of the code does not crash.
    """
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return keypoints, descriptors


def match_landmarks(desc_ref, desc_query):
    """
    Match reference landmarks to query landmarks.

    1. Brute-Force matcher with the L2 (Euclidean) distance.
    2. KNN matching with k = 2: for every reference descriptor we get its
       two closest query descriptors.
    3. Lowe's ratio test: keep a match only if the best distance is clearly
       smaller than the second-best distance. This removes ambiguous matches.

    Returns a list of good cv2.DMatch objects.
    """
    # KNN with k = 2 needs at least 2 descriptors on the query side.
    if len(desc_ref) == 0 or len(desc_query) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc_ref, desc_query, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue                       # not enough neighbours to apply the test
        best, second = pair
        if best.distance < LOWE_RATIO * second.distance:
            good_matches.append(best)

    return good_matches


def decide_recognition(num_good_matches):
    """Return the recognition result as a text label."""
    if num_good_matches >= MIN_GOOD_MATCHES:
        return "OBJECT RECOGNIZED"
    return "OBJECT NOT RECOGNIZED"


def verify_with_homography(kp_ref, kp_query, good_matches):
    """
    Extension: estimate where the object is in the query image.

    cv2.findHomography() with RANSAC finds the 3x3 transform that maps
    reference points onto query points. RANSAC throws away wrong matches
    (outliers) and keeps the consistent ones (inliers).

    Returns (H, inlier_count, mask). H is None if there are not enough matches.
    """
    if len(good_matches) < 4:            # a homography needs at least 4 point pairs
        return None, 0, None

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if H is None or mask is None:
        return None, 0, None

    inliers = int(mask.ravel().sum())
    return H, inliers, mask


def draw_object_outline(query_img, H, ref_shape):
    """
    Project the four corners of the reference image into the query image
    using the homography H, and draw them as a green polygon.
    """
    ref_h, ref_w = ref_shape[:2]
    corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    outlined = query_img.copy()
    cv2.polylines(outlined, [np.int32(projected)], isClosed=True,
                  color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    return outlined


def save_keypoint_image(img, keypoints, path):
    """Draw the detected SIFT landmarks as circles (Task 3: display landmarks)."""
    vis = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(path, vis)
    return vis


def save_match_image(ref_img, kp_ref, query_img, kp_query, good_matches, result_text, path):
    """Draw the good matches side by side and write the result on the image."""
    vis = cv2.drawMatches(ref_img, kp_ref, query_img, kp_query, good_matches, None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    label = f"{result_text}  ({len(good_matches)} good matches)"
    cv2.putText(vis, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(path, vis)
    return vis


def append_observation_row(csv_path, row):
    """Append one run to observations.csv so the observation table can be filled in."""
    header = ["timestamp", "label", "reference_keypoints", "query_keypoints_before_undistortion",
              "query_keypoints_after_undistortion", "good_matches_before_undistortion",
              "good_matches_after_undistortion", "result", "calibration_type"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def show_windows(windows, enabled):
    """Show the result images in windows. Press any key to close them."""
    if not enabled:
        return
    for title, img in windows:
        cv2.imshow(title, img)
    print("[INFO] Press any key while an image window is active to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Object recognition with SIFT landmarks, Lowe ratio test and "
                    "camera-matrix undistortion (OpenCV)."
    )
    parser.add_argument("--reference", required=True,
                        help="Path to the reference image of the object.")
    parser.add_argument("--query", default=None,
                        help="Path to the query image. If omitted, the webcam is used.")
    parser.add_argument("--calib", default=None,
                        help="Path to a .npz calibration file from calibrate_webcam.py. "
                             "If omitted, an APPROXIMATE camera matrix is used.")
    parser.add_argument("--focal", type=float, default=None,
                        help="Approximate focal length in pixels (only used without --calib). "
                             "Default: image width.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index used when --query is not given (default 0).")
    parser.add_argument("--homography", action="store_true",
                        help="Run the extension: RANSAC homography and object outline.")
    parser.add_argument("--label", default="test run",
                        help="Name for this run, written to observations.csv "
                             "(for example 'same object, similar viewpoint').")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Folder where results are saved (default: outputs).")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open windows. Only save the files.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_output_dir(args.output)
    display = not args.no_display
    report = []                       # text lines that are saved to results.txt

    # ------------------------------------------------------------------
    # Task 1: Image acquisition and preprocessing
    # ------------------------------------------------------------------
    print_section("Task 1: Image acquisition and preprocessing")

    ref_img, ref_gray = load_color_and_gray(args.reference)
    print(f"Reference image : {args.reference}  ({ref_img.shape[1]}x{ref_img.shape[0]})")

    if args.query:
        query_img, query_gray = load_color_and_gray(args.query)
        print(f"Query image     : {args.query}  ({query_img.shape[1]}x{query_img.shape[0]})")
    else:
        frame = capture_query_from_webcam(args.camera)
        if frame is None:
            print("[INFO] Capture cancelled. Nothing to do.")
            return
        query_img = frame
        query_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        captured_path = os.path.join(out_dir, "query_captured.jpg")
        cv2.imwrite(captured_path, frame)
        print(f"Query captured from webcam and saved to {captured_path}")

    img_h, img_w = query_img.shape[:2]

    # ------------------------------------------------------------------
    # Task 2: Camera calibration matrix and distortion coefficients
    # ------------------------------------------------------------------
    print_section("Task 2: Camera calibration matrix")

    if args.calib:
        K, dist, calib_size = load_calibration(args.calib)
        K = rescale_camera_matrix(K, calib_size, (img_w, img_h))
        calib_type = f"MEASURED ({os.path.basename(args.calib)})"
    else:
        K, dist = build_approximate_camera_matrix(img_w, img_h, args.focal)
        calib_type = "APPROXIMATE (estimated from image size, not measured)"

    np.set_printoptions(precision=4, suppress=True)
    print("Calibration type:", calib_type)
    print("Camera intrinsic matrix K:")
    print(K)
    print("Distortion coefficients (k1, k2, p1, p2, k3):")
    print(dist.ravel())

    report.append("Calibration type: " + calib_type)
    report.append("Camera matrix K:\n" + np.array2string(K, precision=4, suppress_small=True))
    report.append("Distortion coefficients (k1,k2,p1,p2,k3): " +
                  np.array2string(dist.ravel(), precision=6, suppress_small=True))

    # Correct lens distortion in the query image
    undistorted_img = cv2.undistort(query_img, K, dist)
    undistorted_gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    undistorted_path = os.path.join(out_dir, "undistorted_query.jpg")
    cv2.imwrite(undistorted_path, undistorted_img)
    print(f"Undistorted query saved to {undistorted_path}")

    # ------------------------------------------------------------------
    # Task 3: Landmark detection with SIFT
    # ------------------------------------------------------------------
    print_section("Task 3: Landmark detection (SIFT)")

    sift = cv2.SIFT_create()
    kp_ref, des_ref = detect_landmarks(ref_gray, sift)
    kp_raw, des_raw = detect_landmarks(query_gray, sift)              # before undistortion
    kp_und, des_und = detect_landmarks(undistorted_gray, sift)        # after undistortion

    print(f"Landmarks in reference image              : {len(kp_ref)}")
    print(f"Landmarks in query image (before undist.) : {len(kp_raw)}")
    print(f"Landmarks in query image (after undist.)  : {len(kp_und)}")

    save_keypoint_image(ref_img, kp_ref, os.path.join(out_dir, "keypoints_reference.jpg"))
    save_keypoint_image(undistorted_img, kp_und, os.path.join(out_dir, "keypoints_query_undistorted.jpg"))

    report.append(f"Reference keypoints: {len(kp_ref)}")
    report.append(f"Query keypoints before undistortion: {len(kp_raw)}")
    report.append(f"Query keypoints after undistortion: {len(kp_und)}")

    # ------------------------------------------------------------------
    # Task 4: Landmark matching and object recognition
    # ------------------------------------------------------------------
    print_section("Task 4: Landmark matching and object recognition")

    good_raw = match_landmarks(des_ref, des_raw)      # Task 5 comparison: before undistortion
    good_und = match_landmarks(des_ref, des_und)      # main result: after undistortion

    print(f"Good matches before undistortion: {len(good_raw)}")
    print(f"Good matches after undistortion : {len(good_und)}")

    # The main decision is made on the undistorted query, as the assignment asks.
    result_text = decide_recognition(len(good_und))

    print("\n>>> " + result_text + " <<<")

    match_path = os.path.join(out_dir, "landmark_matches.jpg")
    match_vis = save_match_image(ref_img, kp_ref, undistorted_img, kp_und,
                                 good_und, result_text, match_path)
    print(f"Landmark matching image saved to {match_path}")

    report.append(f"Good matches before undistortion: {len(good_raw)}")
    report.append(f"Good matches after undistortion: {len(good_und)}")
    report.append(f"Recognition result (threshold {MIN_GOOD_MATCHES} good matches): {result_text}")

    windows = [("Reference keypoints", cv2.drawKeypoints(
                    ref_img, kp_ref, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)),
               ("Landmark matches", match_vis)]

    # ------------------------------------------------------------------
    # Extension: homography to verify object location (optional)
    # ------------------------------------------------------------------
    if args.homography:
        print_section("Extension: homography with RANSAC")
        H, inliers, _ = verify_with_homography(kp_ref, kp_und, good_und)
        if H is None:
            print("Homography could not be computed (not enough matches).")
            report.append("Homography: not computed (not enough matches)")
        else:
            verdict = "object location accepted" if inliers >= MIN_HOMOGRAPHY_INLIERS \
                      else "object location rejected (too few inliers)"
            print(f"RANSAC inliers: {inliers}  ->  {verdict}")
            print("Homography matrix H:")
            print(H)
            outlined = draw_object_outline(undistorted_img, H, ref_img.shape)
            outline_path = os.path.join(out_dir, "object_location.jpg")
            cv2.imwrite(outline_path, outlined)
            print(f"Object outline saved to {outline_path}")
            windows.append(("Object location (homography)", outlined))
            report.append(f"Homography RANSAC inliers: {inliers} ({verdict})")

    # ------------------------------------------------------------------
    # Save the report and the observation row (Task 5)
    # ------------------------------------------------------------------
    print_section("Saving results")

    report_path = os.path.join(out_dir, "results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Run label: {args.label}\n")
        f.write(f"Run time : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Reference: {args.reference}\n")
        f.write(f"Query    : {args.query or 'webcam capture'}\n\n")
        f.write("\n".join(report) + "\n")
    print(f"Report saved to {report_path}")

    csv_path = os.path.join(out_dir, "observations.csv")
    append_observation_row(csv_path, [
        datetime.datetime.now().isoformat(timespec="seconds"),
        args.label,
        len(kp_ref),
        len(kp_raw),
        len(kp_und),
        len(good_raw),
        len(good_und),
        result_text,
        calib_type,
    ])
    print(f"Observation row appended to {csv_path}")

    show_windows(windows, display)


if __name__ == "__main__":
    main()
