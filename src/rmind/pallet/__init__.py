"""Monocular pallet detector: keypoints on `cam_fork` + floor-constrained pose.

See `pallet_detection_plan.md` for the design. The modules split as:

    geometry   pallet 3D model, fisheye camera, robot/pallet frames
    pose       keypoints -> pallet (x, y, yaw) in the robot frame
    labels     AprilTag -> pallet pose -> projected keypoint labels
    keypoints  the heatmap keypoint network, its targets and decoding
"""
