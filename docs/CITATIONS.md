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

Used by `face/detect.py` on the left IR image at 848×480. The zoo notes the model detects faces of about 10×10 to 300×300 pixels. A face closer than ~30 cm on this camera is larger than that, so detection also runs at half and quarter size. The 2023mar file has a fixed input shape. It loads under the installed `opencv-python` 5.0.0 and returns no faces on a blank 848×480 image. The zoo also publishes `face_detection_yunet_2026may.onnx` with a dynamic input size for the OpenCV 5 ONNX Runtime engine. This project stays on the 2023mar file while the stream size stays 848×480.

The detector was trained on visible-light faces (WIDER Face). IR is a different image. A miss on a real IR face is a reason to change the detector, not to loosen the depth gate.

## SFace face recogniser

| | |
| --- | --- |
| File | `models/face_recognition_sface_2021dec.onnx` (not in git; `uv run python -m face.models sface`) |
| sha256 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| Source | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), mirrored at [huggingface.co/opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface) |
| License | Apache-2.0 |
| Paper | Yaoyao Zhong, Weihong Deng, Jiani Hu, Dongyue Zhao, Xian Li, Dongchao Wen. *SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition.* IEEE Transactions on Image Processing, 2021. |

Used by `face.embed.SFaceEmbedder` through `cv2.FaceRecognizerSF`, 128-D, on the 112×112 IR crop from `face.align.align_crop` (the same 5-point template). Trained on visible-light faces.

## ArcFace ResNet-50 (InsightFace buffalo_l)

| | |
| --- | --- |
| File | `models/arcface_w600k_r50.onnx` (not in git; `uv run python -m face.models arcface_r50`) |
| sha256 | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| Source | `w600k_r50.onnx` inside [buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip), InsightFace release v0.7 |
| License | InsightFace pretrained models are for non-commercial research only. Accepted for this college project (decision 2026-10-17). |
| Paper | Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019. https://arxiv.org/abs/1801.07698 |
| Training data | WebFace600K, a cleaned subset of WebFace260M (Zhu et al., CVPR 2021) |

Used by `face.embed.ArcFaceEmbedder` through `cv2.dnn`, 512-D, input scaled to [-1, 1]. Runs in about 22 ms per face on this machine.

## AdaFace IR-101 (CVLFace, WebFace12M)

| | |
| --- | --- |
| File | `models/adaface_ir101_webface12m.onnx` (not in git; exported by `uv run --group export python -m face.adaface_export`) |
| Source weights | `model.safetensors` from [minchul/cvlface_adaface_ir101_webface12m](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m) at revision `54f602a0737bd1ee4a4e7e9fd089a485f397fefd`, saved as `models/adaface_ir101_webface12m.safetensors` (`uv run python -m face.models adaface_ir101_weights`) |
| sha256 (weights) | `2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56` |
| License | Code (CVLFace, AdaFace) is MIT. The model card asks users to follow the training data's license; WebFace260M is for non-commercial research. Accepted for this college project, as for ArcFace. |
| Paper | Minchul Kim, Anil K. Jain, Xiaoming Liu. *AdaFace: Quality Adaptive Margin for Face Recognition.* CVPR 2022. https://arxiv.org/abs/2204.00964 |
| Training data | WebFace12M, a subset of WebFace260M (Zhu et al., CVPR 2021) |

No ONNX file is published, so `face/adaface_export.py` rebuilds the IR-101 network locally, loads the weights strictly (every key must match), exports to ONNX, and keeps the file only if OpenCV agrees with PyTorch to within 1e-3 (8.7e-6 on 2026-10-19). The repository's own `trust_remote_code` loader is not used. Used by `face.embed.AdaFaceEmbedder` through `cv2.dnn`, 512-D, RGB scaled to [-1, 1] on the ArcFace 5-point template. The raw output's length is returned as the quality number. Runs in about 42 ms per face on this machine.
