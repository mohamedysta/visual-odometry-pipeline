import cv2
import numpy as np
import glob
import argparse

def calibrate_chessboard(images_glob, board_size=(9,6), square_size=1.0):
    # prepare object points
    objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints, imgpoints = [], []
    images = glob.glob(images_glob)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow('Corners', img); cv2.waitKey(200)
    cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError("No chessboard corners found.")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix:\n", K)
    print("Distortion coeffs:\n", dist.ravel())
    return K, dist

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True, help='glob pattern to chessboard images')
    p.add_argument('--board_w', type=int, default=9)
    p.add_argument('--board_h', type=int, default=6)
    p.add_argument('--square', type=float, default=1.0, help='size of a square in your defined unit')
    args = p.parse_args()

    K, dist = calibrate_chessboard(args.images,
                                   board_size=(args.board_w, args.board_h),
                                   square_size=args.square)
    # Save them for later (e.g. np.savez)
    np.savez('camera_calib.npz', K=K, dist=dist)
    print("Saved camera_calib.npz")