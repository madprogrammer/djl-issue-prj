package com.example;

import ai.djl.Model;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;


public final class ModelTranslator implements Translator<ByteBuffer, DetectedObjects> {
    private Map<Integer, String> classes;
    private final int maxBoxes;
    private final float threshold;

    public ModelTranslator() {
        maxBoxes = 40;
        threshold = 0.7f;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, ByteBuffer input) {
        NDArray imageBytes = ctx.getNDManager().create(input, new Shape(), DataType.STRING);
        imageBytes.setName("image_bytes");

        NDArray key = ctx.getNDManager().create(UUID.randomUUID().toString());
        key.setName("key");

        return new NDList(imageBytes, key);
    }

    @Override
    public void prepare(NDManager manager, Model model) {
        if (classes == null) {
            classes = new ConcurrentHashMap<>();
            classes.put(1, "box");
            classes.put(2, "box_clear");
        }
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        // output of TF object-detection models is a list of tensors, hence NDList in DJL
        // output NDArray order in the list are not guaranteed

        int[] classIds = null;
        float[] probabilities = null;
        NDArray boundingBoxes = null;

        for (NDArray array : list) {
            if (array.getName().equals("detection_boxes")) {
                boundingBoxes = array.get(0);
            } else if (array.getName().equals("detection_scores")) {
                probabilities = array.get(0).toFloatArray();
            } else if (array.getName().equals("detection_classes")) {
                classIds = array.get(0).toType(DataType.INT32, true).toIntArray();
            }
        }

        Objects.requireNonNull(classIds);
        Objects.requireNonNull(probabilities);
        Objects.requireNonNull(boundingBoxes);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        // results are already sorted
        for (int i = 0; i < Math.min(classIds.length, maxBoxes); ++i) {
            int classId = classIds[i];
            double probability = probabilities[i];

            // classId starts from 1, -1 means background
            if (classId > 0 && probability > threshold) {
                String className = classes.getOrDefault(classId, "#" + classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                float yMin = box[0];
                float xMin = box[1];
                float yMax = box[2];
                float xMax = box[3];

                Rectangle rect = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
                retNames.add(className);
                retProbs.add(probability);
                retBB.add(rect);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}