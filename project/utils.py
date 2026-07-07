import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam(model, img_array):
    predictions = model(img_array, training=False)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("No backbone model found inside the loaded model.")
    last_conv_layer = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the backbone model.")
    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = grad_model(
            img_array, training=False
        )
        tape.watch(conv_outputs)
        x = backbone_output
        backbone_found = False
        for layer in model.layers:
            if layer == backbone:
                backbone_found = True
                continue

            if backbone_found:
                x = layer(x, training=False)
        final_predictions = x
        pred_index = tf.argmax(final_predictions[0])
        class_score = final_predictions[:, pred_index]
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    img = img_array[0]
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.uint8(img)
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )
    overlay = cv2.addWeighted(
        img,
        1.0,
        heatmap_color,
        0.4,
        0
    )
    return heatmap_resized, overlay


def generate_explanation(predicted_class):
    explanations = {
        "glioma":
            "The model focused on an abnormal region that matches patterns commonly associated with glioma tumors.",
        "meningioma":
            "The model concentrated on a well-defined mass near the outer brain region, typical of meningioma.",
        "pituitary":
            "The model focused on the lower central brain area where pituitary tumors are commonly located.",
        "notumor":
            "The model did not detect any strong abnormal regions, suggesting normal brain tissue."
    }
    return explanations.get(
        predicted_class.lower(),
        "The highlighted regions most strongly influenced the model's prediction."
    )