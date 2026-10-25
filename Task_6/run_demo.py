#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo.py
===========

Runs the whole assignment with the sample materials, in one command:

  1. Generates the sample images (reference, queries, chessboard set), if missing.
  2. Calibrates the simulated camera from the chessboard images.
  3. Runs object recognition on every test case, with both the approximate and the
     measured camera matrix, and records every run in demo_results/observations.csv.

Works on Windows, macOS and Linux:
    python run_demo.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS = os.path.join(HERE, "sample_materials")
RESULTS = os.path.join(HERE, "demo_results")
PYTHON = sys.executable


def run(args, title):
    """Run one of the assignment scripts and stop with a clear message if it fails."""
    print("\n" + "#" * 70)
    print("# " + title)
    print("#" * 70)
    result = subprocess.run([PYTHON] + args, cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"Step failed: {title}. Read the error above and fix it, then run again.")


def main():
    # Step 1: generate the sample materials if they are not there yet
    if not os.path.exists(os.path.join(MATERIALS, "reference.jpg")):
        run(["make_sample_materials.py", "--out", MATERIALS],
            "Step 1: generate sample images")
    else:
        print("Sample images already exist in sample_materials/. Using them.")

    # Step 2: calibrate the camera from the chessboard images
    run(["calibrate_webcam.py", "--images", os.path.join(MATERIALS, "chessboard"),
         "--rows", "6", "--cols", "9", "--square", "25",
         "--out", os.path.join(MATERIALS, "camera_calib.npz")],
        "Step 2: chessboard calibration")

    calib = os.path.join(MATERIALS, "camera_calib.npz")
    reference = os.path.join(MATERIALS, "reference.jpg")
    queries = [
        ("same object, similar viewpoint", "query_similar_viewpoint.jpg"),
        ("same object, different viewpoint", "query_different_viewpoint.jpg"),
        ("different object", "query_different_object.jpg"),
    ]

    # Step 3: recognition, with approximate and measured calibration
    for label, filename in queries:
        query = os.path.join(MATERIALS, filename)
        run(["object_recognition_calibration.py", "--reference", reference, "--query", query,
             "--label", f"{label} (approximate K)", "--no-display", "--output", RESULTS],
            f"Step 3a: {label}, approximate calibration")
        run(["object_recognition_calibration.py", "--reference", reference, "--query", query,
             "--calib", calib, "--homography",
             "--label", f"{label} (measured K)", "--no-display", "--output", RESULTS],
            f"Step 3b: {label}, measured calibration")

    print("\n" + "=" * 70)
    print("DONE. Results are in demo_results/:")
    print("  landmark_matches.jpg, undistorted_query.jpg, object_location.jpg, results.txt")
    print("  observations.csv (one row per run, the basis of the observation table)")
    print("The files from the last run are kept in demo_results/. Each run overwrites them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
