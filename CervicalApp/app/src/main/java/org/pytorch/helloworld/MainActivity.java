package org.pytorch.helloworld;

import android.content.res.AssetManager;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.media.Image;
import android.media.ImageReader;

import org.checkerframework.common.value.qual.IntVal;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.HexagonDelegate;
import org.pytorch.Device;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.Math;
import java.nio.MappedByteBuffer;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
//import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import androidx.appcompat.app.AppCompatActivity;

//import ai.onnxruntime.OrtEnvironment;
//import ai.onnxruntime.OrtException;
//import ai.onnxruntime.OrtSession;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.flex.FlexDelegate;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class MainActivity extends AppCompatActivity {
  public static float[] MEANNULL = new float[] {0.0f, 0.0f, 0.0f};
  public static float[] STDONE = new float[] {1.0f, 1.0f, 1.0f};

  public static float[] add(float[] first, float[] second) {
    int length = Math.min(first.length, second.length);
    float[] result = new float[length];
    for (int i = 0; i < length; i++) {
      result[i] = first[i] + second[i];
    }
    return result;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmapOrig = null;
    Bitmap bitmap = null;
    Tensor bitmapTensor = null;
    Module module = null;

    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("small_541_image.png"));
      // Resize image
      // bitmap = Bitmap.createScaledBitmap(bitmapOrig, 256, 256, false);
      // Resizing can cause slight error, so I just load in image pre-resized
      bitmapTensor = normalizeImage(bitmap);

      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt

      // cpu -- uncomment one of these
      // for a pytorch test
      module = Module.load(assetFilePath(this, "detr_converted.ptl"));

      // onnx - not working yet
      // OrtSession.SessionOptions session_options = new OrtSession.SessionOptions();
      // session_options.addConfigEntry("session.load_model_format", "ORT");
      //
      // OrtEnvironment env = OrtEnvironment.getEnvironment();
      // OrtSession session = env.createSession(assetFilePath(this,"model.basic.ort"), session_options);

      // vulkan - not working yet
      // module = Module.load ( assetFilePath(this,"dummy_model_vulkan.pt"), null, Device.VULKAN);


    } catch (IOException e) {
      Log.e("CervicalApp", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    int mcIt = 0;
    // Run PyTorch
    long start = System.currentTimeMillis();
    float[] softmaxScores = runPytorchInference(bitmapTensor, module, mcIt);
    long end = System.currentTimeMillis();
    long elapsedTime = (end - start);
    Log.d("DebugStuff", "end: " + elapsedTime);

    // Run TFlite
//    long start = System.currentTimeMillis();
//    float [] softmaxScores = runTFInference(bitmap, mcIt);
//    long end = System.currentTimeMillis();

    // searching for the index with maximum score
//    float maxScore = -Float.MAX_VALUE;
//    int maxScoreIdx = -1;
//    for (int i = 0; i < softmaxScores.length; i++) {
//      System.out.println(softmaxScores[i]);
//      if (softmaxScores[i] > maxScore) {
//        maxScore = softmaxScores[i];
//        maxScoreIdx = i;
//      }
//    }
//
//    Log.d("DebugStuff", "hmm: " + Arrays.toString(softmaxScores));
//
//    String className = CervixClass.CERVIX_CLASSES[maxScoreIdx];
//    long elapsedTime = (end - start);
//
//    // showing className on UI
//    TextView textView = findViewById(R.id.text);
//    String classNameScore = className + " - score:" + maxScore + " - Inference duration: " + elapsedTime + " ms";
//    textView.setText(classNameScore);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  private Tensor normalizeImage(Bitmap bitmap) {
    // ImageNet normalization parameters
    float[] IMAGENET_MEAN = new float[]{0.485f, 0.456f, 0.406f};
    float[] IMAGENET_STD = new float[]{0.229f, 0.224f, 0.225f};

    int width = bitmap.getWidth();
    int height = bitmap.getHeight();

    // Create float array to hold normalized values (RGB channels)
    float[] floatArray = new float[3 * width * height];

    // Get pixels
    int[] pixels = new int[width * height];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

    // Normalize pixels and store directly in float array
    for (int i = 0; i < pixels.length; i++) {
      int pixel = pixels[i];

      // Extract RGB values
      float r = ((pixel >> 16) & 0xFF) / 255.0f;
      float g = ((pixel >> 8) & 0xFF) / 255.0f;
      float b = (pixel & 0xFF) / 255.0f;

      // Apply normalization
      r = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      g = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      b = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

      // Store in float array in CHW format (which PyTorch expects)
      floatArray[i] = r;                    // Red channel
      floatArray[i + width * height] = g;   // Green channel
      floatArray[i + 2 * width * height] = b; // Blue channel

      // Debug first few pixels
      if (i < 10) {
        Log.d("DebugStuff", String.format("Pixel %d: R=%.5f, G=%.5f, B=%.5f", i, r, g, b));
      }
    }

    // Create tensor directly from float array
    long[] shape = new long[]{1, 3, height, width}; // NCHW format
    return Tensor.fromBlob(floatArray, shape);
  }

  private float [] runTFInference(Bitmap bitmapIn, int mcIt) {
    Interpreter interpreter = null;

    try {
      interpreter = getTFInterpreter("modelthirtysix.tflite");
//      interpreter = getTFInterpreter("mobilenet_quant_v1_224.tflite");
//      interpreter = getTFInterpreter("model.tflite");

    } catch (IOException e) {
      e.printStackTrace();
    }

    final int nOutputNeurons = 3;
    final int nBatches = 4;
    float [][] result = new float[nBatches][nOutputNeurons];
    assert interpreter != null;

//    int firstPixelColor = bitmapIn.getPixel(0, 0);
//    int blue  = firstPixelColor & 0xff;
//    Log.d("DebugStuff", "Blue: " + blue);

    ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmapIn, nBatches);
//    TensorImage byteBuffer = convertBitmapToByteBuffer(bitmapIn);
    // Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
//    TensorBuffer probabilityBuffer =
//            TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.UINT8);
    org.tensorflow.lite.Tensor inputDetails = interpreter.getInputTensor(0);
//    Log.d("DebugStuff", "hmm wtf: " + inputDetails.shape().toString());
    int[] myArr = {nBatches, 3, 256, 256};
    interpreter.resizeInput(0, myArr);
    interpreter.run(byteBuffer, result);
//    interpreter.run(byteBuffer.getBuffer(), probabilityBuffer.getBuffer());
    float[] softmaxScores = new float[nOutputNeurons];

    if (mcIt > 0) {
      // Multiple forward passes to simulate MC iterations
      float[] mcScore = new float[nOutputNeurons];
      for (int i = 0; i < mcIt; i++) {
        interpreter.run(byteBuffer, result);
        mcScore = add(mcScore, result[0]);
      }
      // Get mean score
      for (int i = 0; i < nOutputNeurons; i++) {
        softmaxScores[i] = mcScore[i] / mcIt;
      }
    }
    else {
      softmaxScores = result[0];
    }

    return softmaxScores;
  }

  private float [] runPytorchInference(Tensor bitmap, Module module, int mcIt) {
    // final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, MEANNULL, STDONE);
    IValue inputs = IValue.from(bitmap);
    final Map<String, IValue> allOutputs = module.forward(inputs).toDictStringKey();
    // On example image:
    // Should be: [logits, last_hidden_state, pred_boxes, encoder_last_hidden_state]
    Log.d("DebugStuff", allOutputs.keySet().toString());
    // Should be: [0.5416793, 0.52205426, 0.8862307, 0.8713725, 0.6067072, ...]
    Log.d("DebugStuff", "bboxes:  " + Arrays.toString(allOutputs.get("pred_boxes").toTensor().getDataAsFloatArray()));
    // Should be: [[1.7486051, -2.0011644, 2.0146627, -2.3535793, 1.8714539, ...]
    Log.d("DebugStuff", "logits:  " + Arrays.toString(allOutputs.get("logits").toTensor().getDataAsFloatArray()));

    // Define target sizes if needed (height, width pairs)
    // In the example case, we just use 256 the entire way
    // If the original image was 1024x1000 hw, it would be {1024,1000}
    // the 2d shape of the array comes from the fact it can handle batch
    // in this case only one item in batch
    int[][] targetSizes = new int[][]{{256, 256}};

    // Process the detections
    float threshold = 0.0f;
    List<ObjectDetectionPostProcessor.DetectionResult> results = ObjectDetectionPostProcessor.postProcessObjectDetection(
                allOutputs, threshold, targetSizes);

    // Access the results
    float[] scores = null;
    long[] labels = null;
    float[][] boxes = null;
    int[] boxesOut = null;
    for (ObjectDetectionPostProcessor.DetectionResult result : results) {
      scores = result.scores;
      labels = result.labels;
      boxes = result.boxes;
      boxesOut = ObjectDetectionPostProcessor.getBestBoundingBox(result, 256, 256);
    }
    Log.d("DebugStuff", "scores:  " + Arrays.toString(scores));
    Log.d("DebugStuff", "labels:  " + Arrays.toString(labels));
    for (int i = 0; i < boxes.length; i++) {
      Log.d("DebugStuff", String.format("Box %d: [%.2f, %.2f, %.2f, %.2f]",
              i, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]));
    }
    Log.d("DebugStuff", "boxes out:  " + Arrays.toString(boxesOut));

    final int nOutputNeurons = 3;
    float[] softmaxScores = new float[nOutputNeurons];

//    if (mcIt > 0) {
//      // Multiple forward passes to simulate MC iterations
//      float[] mcScore = new float[nOutputNeurons];
//      for (int i = 0; i < mcIt; i++) {
//        float[] score = module.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
//        softmaxScores = softmax(score);
//        mcScore = add(mcScore, softmaxScores);
//      }
//
//      // Get mean score
//      for (int i = 0; i < nOutputNeurons; i++) {
//        softmaxScores[i] = mcScore[i] / mcIt;
//      }
//    }
//    else {
//      // getting tensor content as java array of floats
//      final float[] scores = outputTensor.getDataAsFloatArray();
//      softmaxScores = softmax(scores);
//    }

    return softmaxScores;
  }


  // Stores as C,H,W. To do H,W,C, see original GitHub code
  private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int nBatchSize) {
    ByteBuffer byteBuffer;
    int PIXEL_SIZE = 3; // 3 channels: R, G, B
    int inputSize = 256; // Target size of the image (256x256)
    float IMAGE_MEAN = 0.0f; // Mean for normalization
    float IMAGE_STD = 1.0f;  // Std for normalization

    // Allocate a byte buffer for the data in C, H, W format
    byteBuffer = ByteBuffer.allocateDirect(4 * nBatchSize * PIXEL_SIZE * inputSize * inputSize);
    byteBuffer.order(ByteOrder.nativeOrder());

    // Store pixel data temporarily in this array
    int[] intValues = new int[inputSize * inputSize];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    // Process the bitmap and directly write into ByteBuffer in C, H, W order
    // 1st pass: Write Red channel, 2nd pass: Write Green channel, 3rd pass: Write Blue channel
    // Repeat the process for each batch element
    for (int batch = 0; batch < nBatchSize; ++batch) {
      for (int c = 0; c < PIXEL_SIZE; ++c) { // Iterate over channels: 0 - Red, 1 - Green, 2 - Blue
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
          for (int j = 0; j < inputSize; ++j) {
            final int val = intValues[pixel++];
            float channelValue;

            if (c == 0) {
              // Red channel
              channelValue = (((val >> 16) & 0xFF) / 255.0f - IMAGE_MEAN) / IMAGE_STD;
            } else if (c == 1) {
              // Green channel
              channelValue = (((val >> 8) & 0xFF) / 255.0f - IMAGE_MEAN) / IMAGE_STD;
            } else {
              // Blue channel
              channelValue = ((val & 0xFF) / 255.0f - IMAGE_MEAN) / IMAGE_STD;
            }

            // Write channel value into ByteBuffer
            byteBuffer.putFloat(channelValue);
          }
        }
      }
    }

    return byteBuffer;
  }

//   Stores as C,H,W. To do H,W,C, see original GitHub code
//  private TensorImage convertBitmapToByteBuffer(Bitmap bitmap) {
//    // Initialization code
//    // Create an ImageProcessor with all ops required. For more ops, please
//    // refer to the ImageProcessor Architecture section in this README.
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new NormalizeOp(0, 255))
                    .build();


//  imageProcessor.Builder();
//
//    // Create a TensorImage object. This creates the tensor of the corresponding
//    // tensor type (uint8 in this case) that the LiteRT interpreter needs.
//    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
//
//    // Analysis code for every frame
//    // Preprocess the image
//    tensorImage.load(bitmap);
//    tensorImage = imageProcessor.process(tensorImage);
//    Log.d("DebugStuff", "BEFORE RETURN");
//
//    return tensorImage;
//  }

  private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
    AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public Interpreter getTFInterpreter(String modelPath) throws IOException {
    Interpreter.Options options = new Interpreter.Options();
//
//    // Use NNAPI
//    NnApiDelegate nnApiDelegate = new NnApiDelegate();
//    options.addDelegate(nnApiDelegate);
//
//    // Use GPU, comment if CPU wanted
////    CompatibilityList compatList = new CompatibilityList();
////    GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
////    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
////    options.addDelegate(gpuDelegate);
//
//    // USE CPU, comment if GPU wanted
//    options.setNumThreads(Runtime.getRuntime().availableProcessors());
//
//    return new Interpreter(loadModelFile(this, modelPath), options);
    // Create NNAPI delegate options
//    NnApiDelegate.Options delegateOptions = new NnApiDelegate.Options()
//            .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
//            .setUseNnapiCpu(true)
//            .setAllowFp16(true)
//            .setAcceleratorName("nnapi-reference");

// Pass the options to the NNAPI delegate
//    NnApiDelegate nnApiDelegate = new NnApiDelegate(delegateOptions);
//    NnApiDelegate nnApiDelegate = new NnApiDelegate();

// Add NNAPI delegate to Interpreter options
//    Interpreter.Options options = new Interpreter.Options();
//    options.addDelegate(nnApiDelegate);
    options.setNumThreads(Runtime.getRuntime().availableProcessors()); // Example for setting CPU threads

// Initialize the interpreter with model and options
    return new Interpreter(loadModelFile(this, modelPath), options);
  }

  public float[] softmax(float[] input) {
    float[] exp = new float[input.length];
    float sum = 0;
    for(int neuron = 0; neuron < exp.length; neuron++) {
      double expValue = Math.exp(input[neuron]);
      exp[neuron] = (float)expValue;
      sum += exp[neuron];
    }

    float[] output = new float[input.length];
    for(int neuron = 0; neuron < output.length; neuron++) {
      output[neuron] = exp[neuron] / sum;
    }

    return output;
  }
}