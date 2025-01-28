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

### hpv_16_norm_2.jpg

bboxes: [0.5073654, 0.4913748, 1.0, 0.93179566, 0.5079244, 0.48150685, 1.0, 0.9082691, 0.507653, 0.48980778, 1.0, 0.9203911, 0.50232947, 0.39871106, 0.9893267, 0.7149723, 0.5072057, 0.49175715, 1.0, 0.93539435]

logits: [-3.3264403, 2.753188, -2.9424007, 2.3479314, -3.088187, 2.50523, 3.9038026, -4.244734, -3.3647714, 2.7950597]

scores: [0.0022838004, 0.0050148126, 0.0037084823, 0.99971086, 0.0021081564]

labels: [0, 0, 0, 0, 0]

Box 0: [14.85, 73.37, 2030.85, 2756.95]
Box 1: [15.98, 78.83, 2031.98, 2694.65]
Box 2: [15.43, 85.28, 2031.43, 2736.01]
Box 3: [15.45, 118.73, 2009.94, 2177.85]
Box 4: [14.53, 69.29, 2030.53, 2763.23]

boxes out: [15, 119, 2010, 2178]

### hpv_16_norm_1.jpg

bboxes: [0.50579023, 0.49460772, 1.0, 0.97354287, 0.5059838, 0.49546131, 1.0, 0.9766675, 0.5060617, 0.4951898, 1.0, 0.9770353, 0.46877298, 0.5186536, 0.93087023, 0.9394766, 0.5056768, 0.49429026, 1.0, 0.9717763]

logits: [-2.0843434, 1.6880051, -1.8444053, 1.446581, -1.9642047, 1.5651014, 4.0523224, -3.998963, -2.2335804, 1.8336872]

scores: [0.022480976, 0.03588171, 0.02848979, 0.9996815, 0.016835818]

labels: [0, 0, 0, 0, 0]

Box 0: [11.67, 22.57, 2027.67, 2826.37]
Box 1: [12.06, 20.53, 2028.06, 2833.33]
Box 2: [12.22, 19.22, 2028.22, 2833.08]
Box 3: [6.73, 140.88, 1883.36, 2846.57]
Box 4: [11.44, 24.20, 2027.44, 2822.91]

boxes out: [7, 141, 1883, 2847]

### hpv_16_ind_2.jpg

bboxes: [0.51030535, 0.35240948, 1.0, 0.74973464, 0.50954527, 0.34585127, 1.0, 0.7369905, 0.5095132, 0.34791708, 1.0, 0.7412709, 0.50154555, 0.3416412, 0.9893667, 0.6764206, 0.51008576, 0.35121197, 1.0, 0.7466824]

logits: [0.08210761, -0.39033306, 0.31909356, -0.6409772, 0.37511823, -0.6999395, 3.2511966, -3.4663095, 0.14746705, -0.45590785]

scores: [0.6159613, 0.72313595, 0.74555755, 0.998792, 0.6464281]

labels: [0, 0, 0, 0, 0]

Box 0: [20.78, -64.68, 2036.78, 2094.56]
Box 1: [19.24, -65.21, 2035.24, 2057.32]
Box 2: [19.18, -65.43, 2035.18, 2069.43]
Box 3: [13.83, 9.88, 2008.40, 1957.97]
Box 4: [20.33, -63.73, 2036.33, 2086.71]

boxes out: [14, 10, 2008, 1958]

### hpv_16_ind_1.jng

bboxes: [0.5050558, 0.49303836, 1.0, 0.97202533, 0.504909, 0.4931056, 1.0, 0.9776064, 0.5050447, 0.49299338, 1.0, 0.9750019, 0.51009834, 0.51746386, 0.9647455, 0.8040248, 0.50506246, 0.49307713, 1.0, 0.9697082]

logits: [-3.852434, 3.3183007, -3.5147295, 2.9885223, -3.7880561, 3.2550633, 4.2144265, -4.395959, -3.8945394, 3.3625667]

scores: [7.681674E-4, 0.0014963155, 8.726356E-4, 0.9998179, 7.04649E-4]

labels: [0, 0, 0, 0, 0]

Box 0: [10.19, 20.23, 2026.19, 2819.67]
Box 1: [9.90, 12.39, 2025.90, 2827.90]
Box 2: [10.17, 15.82, 2026.17, 2823.82]
Box 3: [55.89, 332.50, 2000.82, 2648.09]
Box 4: [10.21, 23.68, 2026.21, 2816.44]

boxes out: [56, 333, 2001, 2648]


