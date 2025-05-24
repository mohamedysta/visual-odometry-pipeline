# visual-odometer-for-a-video-recorded-by-your-own-mobile-phone

Estimate camera motion from a monocular phone video using OpenCV.

## Features
- Shi-Tomasi feature detection + Lucas-Kanade optical flow  
- Essential matrix estimation & pose recovery  
- Monocular scale (fixed `ABSOLUTE_SCALE`)  
- Live feature‐track overlay and top-down X-Z trajectory plot  
- Optional camera calibration module

## Quickstart

1. Clone the repo
git clone https://github.com/yourusername/visual-odometry-pipeline.git
cd visual-odometry-pipeline

2. Install dependencies 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


3. (Recommended) Calibrate your camera  
- Use `src/camera_calibration.py` or the Jupyter notebook `notebooks/01_calibration.ipynb`  
- Fill in the resulting `K` and `dist_coeffs` in `vo_pipeline.py`

4. Run visual odometry  
- Via script:
  ```
  bash scripts/run_vo.sh \
    --video path/to/your_video.mp4 \
    --output vo_output_tracked.mp4 \
    --scale 1.0
  ```
- Or directly:
  ```
  python src/vo_pipeline.py \
    --video path/to/your_video.mp4 \
    --output vo_output_tracked.mp4 \
    --scale 1.0
  ```

5. View results  
- Overlaid feature tracks in `vo_output_tracked.mp4`  
- Final trajectory plot `visual_odometry_trajectory.png`

## Repo Layout
├── data/ # sample videos
├── notebooks/ # calibration + run demos
├── scripts/ # convenience bash scripts
└── src/
├── camera_calibration.py
└── vo_pipeline.py

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
