# Cervical Classifier App

```bash
git clone https://github.com/andreanne-lemay/cervical_mobile_app.git
```

## Run the Android App
The Android app `CervicalApp` is based on the demo available [here](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp). In order to preserve the confidentiality of the medical data, a dummy model trained on ImageNet and a dummy image with the same dimensions as one of the medical images available were included in the `assets` folder.

The app does the following steps:
1. Load the image from the assets folder (`dummy_img.jpg`)
2. Crop the image according hard-coded bounding box coordinates which will eventually come from the object detection model.
3. Resize the image to 256x256 to match the processing pipeline done during training.
4. Convert the image to a tensor.
5. Load the PyTorch mobile model (`dummy_model.ptl`).
6. Run inference.

To simulate multiple forward passes describing the Monte Carlo inference approach, the following line in `MainActivity.java` can be modifed.

```java
int mcIt = 0; // Can be set to ~20 to verify MC model runtime. When mcIt is set to 0 it will simply do one forward pass.
```

Using Android studio with a phone emulator or a real device with the app installed, run the app located in the `CervicalApp` folder.

The classified image should appear on the screen. The dummy image is a set of randomly assigned pixel from 0 to 255. The prediction (Normal, Gray zone or precancer), the softmax probability score (from 0.33 to 1) and the inference time will be displayed on the top left. The inference time represents the duration to run the steps 1 to 6 described above. The UI should look like this:

![image](https://user-images.githubusercontent.com/49137243/141835553-6ae9f9b3-1b34-4ef6-a0cd-0000478af618.png)


## Convert PyTorch model to a mobile optimized version
Python version: 3.8.0

Package requirements can be installed using the following CL:
```bash
pip install -r requirements.txt
```

The python script `mobile_model_conversion.py` includes the main steps to convert the PyTorch model into a version optimized and readable by an android app.
Currently, the script takes the ImageNet pretrained weights but the commented lines indicates the step to load the trained model. The script will generate the mobile optimized PyTorch version in the current directory `dummy_model.ptl`.

```bash
python mobile_model_conversion.py
```

## Results on example images:

### hpv16_norm_1_png.png

bboxes: [0.50536674, 0.4911125, 1.0, 0.97616357, 0.50548327, 0.49160314, 1.0, 0.9782266, 0.505567, 0.49128312, 1.0, 0.97894555, 0.47465187, 0.5169286, 0.9447514, 0.95088106, 0.50528973, 0.49109823, 1.0, 0.9735057]

logits: [-2.2319508, 1.8036865, -1.9644412, 1.5305842, -2.0871332, 1.6558794, 4.0526156, -4.0035915, -2.454532, 2.0194993]

scores: [0.01736745, 0.029454105, 0.023134762, 0.999683, 0.011272736]

labels: [0, 0, 0, 0, 0]

Box 0: [10.82, 8.73, 2026.82, 2820.08]
Box 1: [11.05, 7.17, 2027.05, 2824.46]
Box 2: [11.22, 5.21, 2027.22, 2824.58]
Box 3: [4.59, 119.49, 1909.21, 2858.02]
Box 4: [10.66, 12.51, 2026.66, 2816.21]

boxes out: [5, 119, 1909, 2858]

### hpv16_norm_2_png.png

bboxes: [0.50731295, 0.49175832, 1.0, 0.94123846, 0.5083385, 0.4795916, 1.0, 0.906525, 0.50792676, 0.48995182, 1.0, 0.92421025, 0.5023271, 0.39851445, 0.98971015, 0.71798086, 0.5071377, 0.49188656, 1.0, 0.94274837]

logits: [-3.5343297, 2.9524255, -3.118241, 2.493632, -3.2665348, 2.6620753, 3.8917115, -4.2713966, -3.5508037, 2.9730332]

scores: [0.0015211666, 0.0036409132, 0.0026551117, 0.99971503, 0.0014658734]

labels: [0, 0, 0, 0, 0]

Box 0: [14.74, 60.88, 2030.74, 2771.65]
Box 1: [16.81, 75.83, 2032.81, 2686.62]
Box 2: [15.98, 80.20, 2031.98, 2741.92]
Box 3: [15.06, 113.83, 2010.32, 2181.61]
Box 4: [14.39, 59.08, 2030.39, 2774.19]

boxes out: [15, 114, 2010, 2182]

### hpv16_ind_1_png.png

bboxes: [0.50481594, 0.4929444, 1.0, 0.9754277, 0.50479156, 0.49294472, 1.0, 0.97674346, 0.50485253, 0.49293938, 1.0, 0.977247, 0.5084936, 0.5142693, 0.96637493, 0.8153108, 0.5047279, 0.49295387, 1.0, 0.96972555]

logits: [-3.647327, 3.1366289, -3.306315, 2.7926397, -3.6088312, 3.1006663, 4.2467027, -4.416453, -3.7640467, 3.250813]

scores: [0.0011305097, 0.0022401838, 0.0012177918, 0.9998272, 8.9762534E-4]

labels: [0, 0, 0, 0, 0]

Box 0: [9.71, 15.06, 2025.71, 2824.30]
Box 1: [9.66, 13.17, 2025.66, 2826.19]
Box 2: [9.78, 12.43, 2025.78, 2826.90]
Box 3: [51.02, 307.05, 1999.23, 2655.14]
Box 4: [9.53, 23.30, 2025.53, 2816.11]

boxes out: [51, 307, 1999, 2655]

### hpv16_ind_2_png.png

bboxes: [0.5093634, 0.34440807, 1.0, 0.73125106, 0.5085728, 0.33646873, 1.0, 0.7154526, 0.50858694, 0.33877203, 1.0, 0.7199951, 0.5019878, 0.34037402, 0.99018013, 0.6747614, 0.509097, 0.34193584, 1.0, 0.7263806]

logits: [0.2885895, -0.59171546, 0.59205127, -0.90725213, 0.6230676, -0.9325768, 3.174286, -3.3920364, 0.38189492, -0.68525]

scores: [0.70688546, 0.81747055, 0.82572746, 0.99859506, 0.74405354]

labels: [0, 0, 0, 0, 0]

Box 0: [18.88, -61.11, 2034.88, 2044.90]
Box 1: [17.28, -61.22, 2033.28, 1999.28]
Box 2: [17.31, -61.13, 2033.31, 2012.46]
Box 3: [13.91, 8.62, 2010.11, 1951.93]
Box 4: [18.34, -61.21, 2034.34, 2030.76]

boxes out: [14, 9, 2010, 1952]

