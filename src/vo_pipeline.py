import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_camera_calib(calib_file):
    if not os.path.exists(calib_file):
        return None, None
    npz = np.load(calib_file)
    return npz['K'], npz['dist']

def parse_args():
    p = argparse.ArgumentParser(description="Monocular Visual Odometry")
    p.add_argument('--video',    required=True, help='path to input video')
    p.add_argument('--output',   default='vo_output_tracked.mp4', help='file to save tracked video')
    p.add_argument('--calib',    default='camera_calib.npz',   help='npz with K,dist')
    p.add_argument('--scale',    type=float, default=1.0,       help='absolute scale factor')
    p.add_argument('--no_save',  action='store_true',          help='don’t save output video')
    p.add_argument('--no_plot',  action='store_true',          help='disable live plot')
    return p.parse_args()

def main():
    args = parse_args()
    ABSOLUTE_SCALE = args.scale

    # load camera calibration if exists
    K, dist_coeffs = load_camera_calib(args.calib)
    dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4, np.float32)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Could not open {args.video}"

    ret, prev_frame = cap.read()
    assert ret, "Cannot read first frame"
    h, w = prev_frame.shape[:2]

    # fallback K
    if K is None:
        f = max(w, h) * 0.8
        K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], np.float32)
        print("Using approximate K:", K)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300,
                                        qualityLevel=0.01, minDistance=10)

    # VO state
    R_w_c = np.eye(3)
    t_w_c = np.zeros((3,1))
    traj = [t_w_c.flatten()]

    # prep video writer
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc,
                              cap.get(cv2.CAP_PROP_FPS), (w,h))

    # live plot
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([0],[0],'b-')
        ax.set_xlabel("X"), ax.set_ylabel("Z"), ax.grid()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # track
        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        good_new = next_pts[st==1]
        good_old = prev_pts [st==1]

        if len(good_new) >= 8:
            E, mask = cv2.findEssentialMat(good_new, good_old, K,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, good_old, good_new, K, mask=mask)
                Rm = R.T
                tm = -R.T.dot(t)
                t_w_c = t_w_c + R_w_c.dot(ABSOLUTE_SCALE * tm)
                R_w_c = R_w_c.dot(Rm)
                traj.append(t_w_c.flatten())

        # draw
        for (x,y),(u,v) in zip(good_new,good_old):
            cv2.circle(frame, tuple(x.astype(int)), 3, (0,0,255), -1)
            cv2.line(frame, tuple(x.astype(int)), tuple(v.astype(int)), (0,255,0),1)

        cv2.imshow("VO", frame)
        if not args.no_save:
            out.write(frame)

        # plot
        if not args.no_plot and frame_idx%5==0:
            arr = np.array(traj)
            line.set_xdata(arr[:,0]); line.set_ydata(arr[:,2])
            ax.relim(); ax.autoscale()
            fig.canvas.draw(); fig.canvas.flush_events()

        key = cv2.waitKey(1)&0xFF
        if key in (27,ord('q')): break

        prev_gray, prev_pts = gray, good_new.reshape(-1,1,2)
        frame_idx+=1

    cap.release()
    if not args.no_save: out.release()
    cv2.destroyAllWindows()

    # final plot & save
    if not args.no_plot:
        plt.ioff()
        arr = np.array(traj)
        plt.figure(); plt.plot(arr[:,0],arr[:,2],'b-')
        plt.xlabel("X"); plt.ylabel("Z"); plt.grid()
        plt.savefig("visual_odometry_trajectory.png")
        plt.show()

if __name__=="__main__":
    main()