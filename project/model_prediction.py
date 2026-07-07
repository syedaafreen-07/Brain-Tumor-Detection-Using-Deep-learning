
def show_model_prediction():
    import streamlit as st
    import os
    from tensorflow.keras.models import load_model
    from keras.preprocessing.image import load_img, img_to_array
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    import streamlit as st
    from utils import generate_gradcam, generate_explanation

    IMG_SIZE = (224, 224)

    
    MODEL_PATHS = {
        "Xception": "models/Xception_model.keras",
        "ResNet50": "models/ResNet50_model.keras",
        "VGG16": "models/VGG16_model.keras",
        "EfficientNetB0": "models/EfficientNetB0_model.keras"
    }

   
    CLASS_NAMES = ["No Tumor", "Tumor"]
    best_model = "Xception"
    
    @st.cache_resource
    def load_best_model():
        model_path = MODEL_PATHS[best_model]
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()
        return load_model(model_path)
    
    
    class_labels = ['glioma', 'notumor', 'meningioma', 'pituitary']

    def detect_tumor_streamlit(image, model, image_size=128):
        try: 
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predict = model.predict(img_array)
            predicted_class_index = np.argmax(predict, axis=1)[0]
            confidence_score = float(np.max(predict, axis=1)[0])
            predicted_label = class_labels[predicted_class_index]
            if predicted_label == 'notumor':
                result = "No Tumor"
            else:
                result = f"Tumor: {predicted_label.capitalize()}"
            probabilities = {
                class_labels[i]: float(predict[0][i])
                for i in range(len(class_labels))
            }
            
            return result, confidence_score, probabilities,img_array
        except Exception as e:
            return f"Error: {str(e)}", 0.0, {}
    
    st.title("🔍 Brain Tumor Prediction")
    st.markdown(
        f"Using the best model: **{best_model}**"
    )
    uploaded_file = st.file_uploader(
        "Upload an MRI image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        with st.spinner("Loading model and predicting..."):
            model = load_best_model()
            predicted_class, confidence, probabilities,img_array = detect_tumor_streamlit(
                image,model
            )

        
        st.subheader("Prediction Result")
        if predicted_class.lower() == "tumor":
                st.error(
                    f"🧠 Prediction: {predicted_class}\n\n"
                    f"Confidence: {confidence:.2%}"
                )
        else:
                st.success(
                    f"✅ Prediction: {predicted_class}\n\n"
                    f"Confidence: {confidence:.2%}"
                )
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
                {
                    "Class": list(probabilities.keys()),
                    "Probability": list(probabilities.values())
                }
        )
        st.dataframe(prob_df, use_container_width=True)
        st.bar_chart(
                prob_df.set_index("Class")["Probability"]
        )
        heatmap, overlay = generate_gradcam(model, img_array)
        explanation = generate_explanation(predicted_class)
        st.image(overlay, caption="Grad-CAM Heatmap",width=350)
        st.info(explanation)
    else:
        st.info("Please upload an MRI image to begin prediction.")