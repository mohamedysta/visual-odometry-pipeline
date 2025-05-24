import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
VIDEO_PATH = r"C:\\Users\\Abd Elrahman\\Videos\\video.mp4"
  # <--- !!! REPLACE WITH YOUR VIDEO FILE PATH !!!
# Example: VIDEO_PATH = "test_video.mp4" # If you have a video named test_video.mp4 in the same directory

OUTPUT_VIDEO_FILE = "vo_output_tracked.mp4" # File to save video with tracked features
PLOT_TRAJECTORY = True
   # Set to False if you don't want to save the processed video

# --- Camera Parameters (CRUCIAL: Calibrate your camera for good results!) ---
# If you have calibrated your camera, uncomment and fill these:
# K = np.array([[fx,  0, cx],
#               [ 0, fy, cy],
#               [ 0,  0,  1]], dtype=np.float32)
# dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32) # Distortion coefficients

# If K is None, it will be approximated from the first frame's dimensions.
K = None
dist_coeffs = np.zeros(4, dtype=np.float32) # Assuming no distortion if not calibrated

# --- Feature Detection and Tracking Parameters ---
# Parameters for Shi-Tomasi corner detection (goodFeaturesToTrack)
feature_params = dict(maxCorners=300,       # Max number of corners to return
                      qualityLevel=0.01,    # Minimal accepted quality of image corners (0-1)
                      minDistance=10,       # Minimum possible Euclidean distance between corners
                      blockSize=7)          # Size of an average block for computing derivative covariation matrix

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21),          # size of the search window at each pyramid level
                 maxLevel=3,                # 0-based maximal pyramid level number
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

MIN_FEATURES_FOR_MOTION = 8     # Minimum number of tracked features to attempt motion estimation
MIN_FEATURES_TO_REDETECT = 50   # If tracked features fall below this, re-detect all features

# --- Scale Parameter (Monocular VO Challenge) ---
ABSOLUTE_SCALE = 1.0  # This is a fixed scale factor. Its value affects the "size" of the trajectory.
                      # In a real system, this needs to be estimated or known.

# --- Trajectory State ---
# current_R_w_c: Rotation of the camera in the world frame (World FROM Camera)
# current_t_w_c: Translation of the camera origin in the world frame
current_R_w_c = np.eye(3)
current_t_w_c = np.zeros((3, 1))
trajectory_points = [current_t_w_c.flatten().copy()] # Store (x, y, z) positions

def main():
    global K, current_R_w_c, current_t_w_c, trajectory_points, SAVE_OUTPUT_VIDEO

    SAVE_OUTPUT_VIDEO = True   
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        print("Please check the VIDEO_PATH variable in the script.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from video.")
        cap.release()
        return

    if K is None:
        h, w = prev_frame.shape[:2]
        focal_length_approx = max(w, h) * 0.8 # A common heuristic, adjust if needed
        K = np.array([[focal_length_approx, 0, w/2.0],
                      [0, focal_length_approx, h/2.0],
                      [0, 0, 1.0]], dtype=np.float32)
        print(f"Warning: Using approximate K based on image dimensions ({w}x{h}): \n{K}")
        print("For accurate results, please calibrate your camera and set K and dist_coeffs explicitly.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if prev_points is None or len(prev_points) < MIN_FEATURES_FOR_MOTION:
        print("Error: Not enough features found in the first frame. Try adjusting feature_params or check video quality.")
        cap.release()
        return
    
    mask_draw = np.zeros_like(prev_frame) # Mask for drawing feature tracks

    frame_idx = 0
    
    video_out = None

    if SAVE_OUTPUT_VIDEO:
        h_out, w_out = prev_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, cap.get(cv2.CAP_PROP_FPS), (w_out, h_out))
        if not video_out.isOpened():
            print(f"Error: Could not open video writer for {OUTPUT_VIDEO_FILE}")
            SAVE_OUTPUT_VIDEO = False

    print(f"Starting Visual Odometry. Press 'ESC' or 'q' to quit.")
    
    if PLOT_TRAJECTORY:
        plt.ion() 
        fig_traj, ax_traj = plt.subplots(figsize=(8,6))
        line_traj, = ax_traj.plot([0], [0], 'bo-', markersize=3) 
        ax_traj.set_xlabel("X (world)")
        ax_traj.set_ylabel("Z (world)") # Typically, Z is forward in camera coordinates
        ax_traj.set_title("Camera Trajectory (Top-Down View: X-Z plane)")
        ax_traj.axis('equal') 
        ax_traj.grid(True)
        plt.show()

    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("End of video.")
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        vis_frame = current_frame.copy() 

        if prev_points is None or len(prev_points) < MIN_FEATURES_TO_REDETECT:
            print(f"Frame {frame_idx}: Low feature count ({len(prev_points) if prev_points is not None else 0}). Re-detecting features.")
            prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            mask_draw = np.zeros_like(prev_frame) # Reset drawing mask
            if prev_points is None or len(prev_points) < MIN_FEATURES_FOR_MOTION:
                print(f"Frame {frame_idx}: Re-detection failed. Skipping VO for this frame.")
                prev_gray = current_gray.copy()
                prev_points = None
                if SAVE_OUTPUT_VIDEO and video_out: video_out.write(vis_frame)
                cv2.imshow("Visual Odometry - Features", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break
                frame_idx += 1
                continue

        current_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None, **lk_params)

        if current_points is not None and status is not None:
            good_new_pts = current_points[status.ravel() == 1]
            good_old_pts = prev_points[status.ravel() == 1]
        else:
            good_new_pts, good_old_pts = np.array([]), np.array([])

        if len(good_new_pts) >= MIN_FEATURES_FOR_MOTION:
            E, mask_e = cv2.findEssentialMat(good_new_pts, good_old_pts, cameraMatrix=K,
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and mask_e is not None:
                num_inliers, R_rel_curr_prev, t_rel_curr_prev_unit, mask_rp = cv2.recoverPose(
                                                                    E, good_old_pts, good_new_pts, 
                                                                    cameraMatrix=K, mask=mask_e)

                if num_inliers > MIN_FEATURES_FOR_MOTION / 2 and R_rel_curr_prev is not None and t_rel_curr_prev_unit is not None:
                    # R_rel_curr_prev, t_rel_curr_prev_unit is the transform from PREVIOUS to CURRENT camera frame
                    # P_current = R_rel_curr_prev * P_previous + t_rel_curr_prev_unit
                    
                    # We want T_world_current = T_world_previous * T_previous_current
                    # T_previous_current is the inverse of T_current_previous
                    # R_prev_curr = R_rel_curr_prev.T
                    # t_prev_curr_in_prev_coords = -R_rel_curr_prev.T @ t_rel_curr_prev_unit

                    R_motion = R_rel_curr_prev.T 
                    t_motion_in_prev_frame = -R_rel_curr_prev.T @ t_rel_curr_prev_unit
                    
                    # Apply the transformation in the world frame
                    current_t_w_c = current_t_w_c + current_R_w_c @ (ABSOLUTE_SCALE * t_motion_in_prev_frame)
                    current_R_w_c = current_R_w_c @ R_motion
                    
                    trajectory_points.append(current_t_w_c.flatten().copy())
                else:
                    print(f"Frame {frame_idx}: recoverPose failed or low inliers ({num_inliers}).")
            else:
                print(f"Frame {frame_idx}: findEssentialMat failed.")
        else:
            print(f"Frame {frame_idx}: Not enough good tracked points ({len(good_new_pts)}).")

        # --- Visualization ---
        for i, (new, old) in enumerate(zip(good_new_pts, good_old_pts)): # Use only successfully tracked points for drawing
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask_draw = cv2.line(mask_draw, (a, b), (c, d), (0, 255, 0), 1) # Green tracks
            vis_frame = cv2.circle(vis_frame, (a, b), 3, (0, 0, 255), -1)   # Red circles at current points
        
        img_with_tracks = cv2.add(vis_frame, mask_draw)
        cv2.putText(img_with_tracks, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img_with_tracks, f"Tracked: {len(good_new_pts)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow("Visual Odometry - Features", img_with_tracks)
        if SAVE_OUTPUT_VIDEO and video_out:
            video_out.write(img_with_tracks)

        # --- Update for next iteration ---
        prev_gray = current_gray.copy()
        prev_points = good_new_pts.reshape(-1, 1, 2) if len(good_new_pts) > 0 else None # Use good new points as old points for next frame

        # --- Update Trajectory Plot ---
        if PLOT_TRAJECTORY and frame_idx % 5 == 0: # Update plot every 5 frames for performance
            if len(trajectory_points) > 0:
                traj_array = np.array(trajectory_points)
                # OpenCV camera: +X right, +Y down, +Z forward.
                # For top-down (X-Z plane):
                line_traj.set_xdata(traj_array[:, 0]) # X coordinates
                line_traj.set_ydata(traj_array[:, 2]) # Z coordinates
                ax_traj.relim()      # Recompute the data limits
                ax_traj.autoscale_view() # Autoscale the view
                fig_traj.canvas.draw()
                fig_traj.canvas.flush_events()

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF # Wait for 1 ms (or longer if processing is slow)
        if key == 27 or key == ord('q'): # ESC or 'q'
            print("Exiting...")
            break
            
    # --- Cleanup ---
    cap.release()
    if SAVE_OUTPUT_VIDEO and video_out:
        video_out.release()
    cv2.destroyAllWindows()

    if PLOT_TRAJECTORY:
        plt.ioff() # Turn off interactive mode
        if len(trajectory_points) > 0:
            traj_array = np.array(trajectory_points)
            fig_final, ax_final = plt.subplots(figsize=(10,8))
            ax_final.plot(traj_array[:, 0], traj_array[:, 2], 'bo-', label="Estimated Trajectory (X-Z)") # X-Z for top-down
            ax_final.set_xlabel("X (world)")
            ax_final.set_ylabel("Z (world)")
            ax_final.set_title(f"Final Camera Trajectory (Top-Down View, {frame_idx} frames)")
            ax_final.axis('equal')
            ax_final.grid(True)
            ax_final.legend()
            plt.savefig("visual_odometry_trajectory.png")
            print("Trajectory plot saved as visual_odometry_trajectory.png")
            plt.show() # Keep the plot window open until manually closed
        else:
            print("No trajectory points to plot.")

if __name__ == '__main__':

    # --- How to use ---
    # 1. Replace "your_mobile_video.mp4" with the actual path to your video file.
    # 2. (Highly Recommended) Calibrate your camera:
    #    - Take several pictures of a chessboard pattern with your phone.
    #    - Use OpenCV's camera calibration tools (e.g., cv2.calibrateCamera()) to find
    #      the camera matrix K and distortion coefficients dist_coeffs.
    #    - Update the K and dist_coeffs variables in this script.
    #    - If you don't do this, a rough approximation for K will be used,
    #      and distortion will be ignored, leading to less accurate results.
    # 3. Run the script: python your_script_name.py (replace your_script_name.py with the actual filename)
    #
    # Notes on ABSOLUTE_SCALE:
    # - The trajectory's size is determined by this.
    # - To get a metric scale (i.e., trajectory in meters), you'd typically need:
    #   a) Ground truth distance for some segment of the video to calibrate the scale.
    #   b) Another sensor (e.g., IMU providing scaled acceleration, GPS).
    #   c) Known object sizes in the scene for triangulation and scale determination.
    # - For this basic example, ABSOLUTE_SCALE = 1.0 means we are accumulating
    #   the unit translations directly. You can experiment with this value.
    
    main()