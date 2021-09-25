/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.example;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Paths;

@Path("/")
public class ImageClassification {

    private static final String IMAGE_URL = "https://www.anufrienko.net/files/djltest.jpg";

    @Path("/detect")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String detect() throws TranslateException, IOException, ModelException {
        URL url = new URL(IMAGE_URL);
        ByteBuffer buffer;

        try (InputStream is = url.openStream()) {
            buffer = ByteBuffer.wrap(is.readAllBytes());
        }

        System.out.println(Paths.get(System.getProperty("user.dir"), "..", "tensorflow"));
        Criteria<ByteBuffer, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(ByteBuffer.class, DetectedObjects.class)
                .optTranslator(new ModelTranslator())
                .optEngine("TensorFlow")
                .optDevice(Device.cpu())
                .optModelPath(Paths.get(System.getProperty("user.dir"), "..", "tensorflow"))
                .build();

        try (ZooModel<ByteBuffer, DetectedObjects> model = criteria.loadModel();
             Predictor<ByteBuffer, DetectedObjects> predictor = model.newPredictor()) {
            Classifications result = predictor.predict(buffer);
            return result.toString() + "\n";
        }
    }
}
