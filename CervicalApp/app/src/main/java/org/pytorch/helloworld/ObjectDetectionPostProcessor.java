package org.pytorch.helloworld;

import org.pytorch.IValue;
import org.pytorch.Tensor;
import java.util.*;

public class ObjectDetectionPostProcessor {

    public static class DetectionResult {
        public final float[] scores;
        public final long[] labels;
        public final float[][] boxes;

        public DetectionResult(float[] scores, long[] labels, float[][] boxes) {
            this.scores = scores;
            this.labels = labels;
            this.boxes = boxes;
        }
    }


    /**
     * Processes object detection model outputs to extract and format detection results.
     * @param outputs Model result output containing logits and predicted boxes
     * @param threshold Confidence threshold for filtering detections. Should be 0 usually
     * @param targetSizes Original image dimensions for scaling boxes [[height, width], ...]
     * @return List of DetectionResult objects containing filtered and processed detections
     */
    public static List<DetectionResult> postProcessObjectDetection(
            Map<String, IValue> outputs,
            float threshold,
            int[][] targetSizes) {

        // Extract logits and pred_boxes from outputs
        Tensor logitsTensor = outputs.get("logits").toTensor();
        Tensor predBoxesTensor = outputs.get("pred_boxes").toTensor();

        // Get tensor shapes
        long[] logitsShape = logitsTensor.shape();
        long batchSize = logitsShape[0];
        long numQueries = logitsShape[1];
        long numClasses = logitsShape[2];

        // Convert tensors to arrays for processing
        float[][][] logits = tensorToArray3d(logitsTensor);
        float[][][] predBoxes = tensorToArray3d(predBoxesTensor);

        List<DetectionResult> results = new ArrayList<>();

        // Process each image in the batch
        for (int i = 0; i < batchSize; i++) {
            // Apply softmax and get max scores and labels
            float[][] scores = new float[(int)numQueries][(int)(numClasses - 1)];
            long[] labels = new long[(int)numQueries];
            float[] maxScores = new float[(int)numQueries];

            for (int j = 0; j < numQueries; j++) {
                float[] classProbabilities = softmax(logits[i][j]);
                float maxScore = Float.NEGATIVE_INFINITY;
                int maxLabel = 0;

                // Skip last class (background) in probabilities
                for (int k = 0; k < numClasses - 1; k++) {
                    if (classProbabilities[k] > maxScore) {
                        maxScore = classProbabilities[k];
                        maxLabel = k;
                    }
                }

                maxScores[j] = maxScore;
                labels[j] = maxLabel;
            }

            // Convert boxes from center format to corner format
            float[][] boxes = new float[(int)numQueries][4];
            for (int j = 0; j < numQueries; j++) {
                float[] centerBox = predBoxes[i][j];
                boxes[j] = centerToCornersFormat(centerBox);
            }

            // Scale boxes if target sizes are provided
            if (targetSizes != null) {
                float[] scaleFactor = new float[]{
                        targetSizes[i][1], targetSizes[i][0],  // width, height
                        targetSizes[i][1], targetSizes[i][0]   // width, height
                };

                for (int j = 0; j < numQueries; j++) {
                    for (int k = 0; k < 4; k++) {
                        boxes[j][k] *= scaleFactor[k];
                    }
                }
            }

            // Filter by threshold
            List<Float> filteredScores = new ArrayList<>();
            List<Long> filteredLabels = new ArrayList<>();
            List<float[]> filteredBoxes = new ArrayList<>();

            for (int j = 0; j < numQueries; j++) {
                if (maxScores[j] > threshold) {
                    filteredScores.add(maxScores[j]);
                    filteredLabels.add(labels[j]);
                    filteredBoxes.add(boxes[j]);
                }
            }

            // Convert Lists to arrays
            float[] resultScores = new float[filteredScores.size()];
            long[] resultLabels = new long[filteredLabels.size()];
            float[][] resultBoxes = new float[filteredBoxes.size()][4];

            for (int j = 0; j < filteredScores.size(); j++) {
                resultScores[j] = filteredScores.get(j);
                resultLabels[j] = filteredLabels.get(j);
                resultBoxes[j] = filteredBoxes.get(j);
            }

            results.add(new DetectionResult(resultScores, resultLabels, resultBoxes));
        }

        return results;
    }

    /**
     * Applies softmax function to normalize input array into probabilities.
     * Uses numerical stability optimization by subtracting max value.
     * @param input Array of raw logits
     * @return Array of probability scores summing to 1.0
     */
    private static float[] softmax(float[] input) {
        float[] output = new float[input.length];
        float max = Float.NEGATIVE_INFINITY;

        // Find max for numerical stability
        for (float value : input) {
            if (value > max) {
                max = value;
            }
        }

        float sum = 0;
        // Compute exp(x - max) and sum
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i] - max);
            sum += output[i];
        }

        // Normalize
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    /**
     * Converts bounding box from center format (x, y, w, h) to corner format (x1, y1, x2, y2).
     * @param centerBox Array containing [center_x, center_y, width, height]
     * @return Array containing [x1, y1, x2, y2] corner coordinates
     */
    private static float[] centerToCornersFormat(float[] centerBox) {
        float centerX = centerBox[0];
        float centerY = centerBox[1];
        float width = centerBox[2];
        float height = centerBox[3];

        return new float[]{
                centerX - width/2,  // x0
                centerY - height/2, // y0
                centerX + width/2,  // x1
                centerY + height/2  // y1
        };
    }

    /**
     * Converts 3D PyTorch tensor to native Java 3D array.
     * @param tensor Input tensor with shape [batch_size, num_queries, dim]
     * @return Equivalent 3D float array with same dimensions
     */
    private static float[][][] tensorToArray3d(Tensor tensor) {
        long[] shape = tensor.shape();
        float[][][] array = new float[(int)shape[0]][(int)shape[1]][(int)shape[2]];
        float[] flatArray = tensor.getDataAsFloatArray();

        int index = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    array[i][j][k] = flatArray[index++];
                }
            }
        }

        return array;
    }

    /**
     * Gets the highest confidence bounding box from detection results and clips to image dimensions.
     * @param result DetectionResult containing scores and boxes (and labels but only one class here)
     * @param width Image width for clipping bounds
     * @param height Image height for clipping bounds
     * @return Array containing [x1, y1, x2, y2] of best detection clipped to image bounds
     *
     */
    public static int[] getBestBoundingBox(ObjectDetectionPostProcessor.DetectionResult result, int width, int height) {
        int bestIdx = 0;
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < result.scores.length; i++) {
            if (result.scores[i] > maxScore) {
                maxScore = result.scores[i];
                bestIdx = i;
            }
        }

        int x1 = Math.round(result.boxes[bestIdx][0]);
        int y1 = Math.round(result.boxes[bestIdx][1]);
        int x2 = Math.round(result.boxes[bestIdx][2]);
        int y2 = Math.round(result.boxes[bestIdx][3]);

        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        x2 = Math.min(width, x2);
        y2 = Math.min(height, y2);

        return new int[]{x1, y1, x2, y2};
    }
}