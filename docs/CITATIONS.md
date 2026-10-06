# Citations

External models used by this project. Do not add a dataset here until one is actually used.

## YuNet face detector

| | |
| --- | --- |
| File | `models/face_detection_yunet_2023mar.onnx` |
| sha256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Source | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), file `face_detection_yunet_2023mar.onnx` |
| Upstream weights | [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train/blob/a61a428929148171b488f024b5d6774f93cdbc13/tasks/task1/onnx/yunet.onnx) |
| License | OpenCV Zoo is Apache-2.0. The zoo redistributes this file. |
| Paper | Wei Wu, Hanyang Peng, Shiqi Yu. *Yunet: A tiny millisecond-level face detector.* Machine Intelligence Research 20, 656–665 (2023). https://doi.org/10.1007/s11633-023-1423-y |

Used by `face/detect.py` on the left IR image at 848×480. The 2023mar file has a fixed input shape. It loads under the installed `opencv-python` 5.0.0 and returns no faces on a blank 848×480 image. The zoo also publishes `face_detection_yunet_2026may.onnx` with a dynamic input size for the OpenCV 5 ONNX Runtime engine. This project stays on the 2023mar file while the stream size stays 848×480.

The detector was trained on visible-light faces (WIDER Face). IR is a different image. A miss on a real IR face is a reason to change the detector, not to loosen the depth gate.
