import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64

# --- CONFIGURATION ---
MODEL_PATH = 'multi_disease_lungs_model_final.keras'
IMG_SIZE = (150, 150)
# The name of the inner model layer (from your model summary)
BASE_LAYER_NAME = 'efficientnetb0' 

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

DISEASE_DETAILS = {
    'Atelectasis': 'Partial or complete collapse of the lung.',
    'Cardiomegaly': 'Enlarged heart, often a sign of heart failure.',
    'Consolidation': 'Air in the lung is replaced by fluid or solid (e.g., pneumonia).',
    'Edema': 'Excess fluid in the lungs, often caused by heart problems.',
    'Effusion': 'Buildup of fluid between the layers of tissue that line the lungs.',
    'Emphysema': 'Damage to the air sacs (alveoli), causing shortness of breath.',
    'Fibrosis': 'Scarring of lung tissue, making it thick and stiff.',
    'Hernia': 'Organ pushing through an opening in the muscle or tissue.',
    'Infiltration': 'A substance (pus, blood, protein) lingering in the lung tissue.',
    'Mass': 'A large abnormal growth (>3cm), potentially a tumor.',
    'Nodule': 'A small abnormal growth (<3cm), often called a spot on the lung.',
    'Pleural_Thickening': 'Thickening of the lining of the lungs (pleura).',
    'Pneumonia': 'Infection that inflames the air sacs in one or both lungs.',
    'Pneumothorax': 'Collapsed lung caused by air leaking into the space between lung and chest wall.'
}

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model = None
conv_model = None
classifier_layers = []

print("Loading model... please wait.")
try:
    model = load_model(MODEL_PATH)
    print("✅ Main Model loaded successfully!")

    # --- SETUP GRAD-CAM (MANUAL HEAD METHOD) ---
    try:
        # 1. Get the nested Base Model (EfficientNet)
        base_layer = model.get_layer(BASE_LAYER_NAME)
        
        # 2. Get the last conv layer *inside* the base model
        # 'top_activation' is the standard output for EfficientNet
        last_conv_layer = base_layer.get_layer('top_activation')
        
        # 3. Create a model that gives us the Conv Features
        conv_model = Model(base_layer.inputs, last_conv_layer.output)
        
        # 4. Identify the "Head" layers (everything AFTER the base)
        # We scan the main model's layers to find where the base ends
        base_index = None
        for i, layer in enumerate(model.layers):
            if layer.name == BASE_LAYER_NAME:
                base_index = i
                break
        
        # Store the remaining layers (Pooling, Dropout, Dense) to run manually later
        if base_index is not None:
            classifier_layers = model.layers[base_index+1:]
            print(f"✅ Grad-CAM setup complete. Found {len(classifier_layers)} head layers.")
        else:
            print("⚠️ Warning: Could not find base layer in model.")

    except Exception as setup_err:
        print(f"⚠️ Warning: Grad-CAM setup failed: {setup_err}")

except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==========================================
# GRAD-CAM FUNCTIONS
# ==========================================

def make_gradcam_heatmap(img_array, pred_index=None):
    """ Generates a heatmap using the Manual Head approach. """
    if conv_model is None:
        raise ValueError("Grad-CAM not initialized.")

    with tf.GradientTape() as tape:
        # 1. Get Features from Base Model
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs = conv_model(inputs)
        
        # 2. Watch the features
        tape.watch(conv_outputs)
        
        # 3. Manually pass features through the Head Layers
        x = conv_outputs
        for layer in classifier_layers:
            x = layer(x)
        
        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 4. Calculate Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    # 5. Pool Gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 6. Generate Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    buffer = io.BytesIO()
    superimposed_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ==========================================
# FLASK ROUTES
# ==========================================

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    temp_path = "temp_upload.png"
    file.save(temp_path)
    
    try:
        # 1. Process image
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        processed_image = prepare_image(img_bytes)
        
        # 2. Predict
        logits = model.predict(processed_image)
        probabilities = tf.nn.sigmoid(logits).numpy()[0]
        
        # 3. Get Top Prediction
        top_pred_index = np.argmax(probabilities)
        top_pred_label = CLASSES[top_pred_index]
        
        # 4. Grad-CAM
        heatmap_image_b64 = None
        try:
            if conv_model:
                heatmap = make_gradcam_heatmap(processed_image, pred_index=top_pred_index)
                heatmap_image_b64 = overlay_heatmap(temp_path, heatmap)
        except Exception as grad_err:
            print(f"Grad-CAM failed: {grad_err}")

        # 5. Format Results
        results_list = []
        for i, label in enumerate(CLASSES):
            score_percent = float(probabilities[i]) * 100
            results_list.append({
                "label": label,
                "score": round(score_percent, 2),
                "description": DISEASE_DETAILS.get(label, "No description available.")
            })
            
        results_list.sort(key=lambda x: x["score"], reverse=True)
        max_score = results_list[0]["score"]
        
        return jsonify({
            'status': 'success',
            'predictions': results_list,
            'max_score': max_score,
            'gradcam_image': heatmap_image_b64,
            'top_label': top_pred_label
        })

    except Exception as e:
        print(f"❌ CRITICAL SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)