"""
Leaf Disease Detection Web Application
A modern Flask-based web interface for plant disease detection
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import io
import base64

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
class_indices = None
disease_info = None

# Disease information database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis that affects apple trees.",
        "symptoms": "Dark, olive-green to brown spots on leaves and fruit. Leaves may curl and drop early.",
        "treatment": "Apply fungicides during spring. Remove infected leaves and fruit. Improve air circulation.",
        "prevention": "Plant resistant varieties. Rake and destroy fallen leaves. Prune trees for better airflow."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Botryosphaeria obtusa affecting fruit, leaves, and bark.",
        "symptoms": "Brown spots with concentric rings on leaves. Rotting fruit with black, mummified appearance.",
        "treatment": "Remove infected plant parts. Apply fungicides during growing season.",
        "prevention": "Maintain tree health. Remove mummified fruits. Prune dead wood."
    },
    "Apple___Brown spot": {
        "plant": "Apple",
        "disease": "Brown Spot",
        "description": "A fungal infection causing brown lesions on apple leaves.",
        "symptoms": "Brown circular spots on leaves, potentially leading to early defoliation.",
        "treatment": "Apply appropriate fungicides. Remove affected foliage.",
        "prevention": "Ensure good air circulation. Avoid overhead watering."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar/juniper trees to complete its life cycle.",
        "symptoms": "Yellow-orange spots on leaves with small black dots. Tube-like structures on leaf undersides.",
        "treatment": "Apply fungicides in spring. Remove nearby cedar/juniper trees if possible.",
        "prevention": "Plant resistant varieties. Remove galls from cedars before spring."
    },
    "Apple___Grey spot": {
        "plant": "Apple",
        "disease": "Grey Spot",
        "description": "A fungal disease affecting apple leaves causing grey lesions.",
        "symptoms": "Grey to brown spots on leaves, may cause premature leaf drop.",
        "treatment": "Apply fungicides. Improve orchard sanitation.",
        "prevention": "Remove fallen leaves. Ensure proper spacing between trees."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "This apple leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present. Leaf appears healthy with normal coloration.",
        "treatment": "No treatment needed. Continue regular plant care.",
        "prevention": "Maintain good growing conditions and regular monitoring."
    },
    "Apple___Mosaic": {
        "plant": "Apple",
        "disease": "Mosaic Virus",
        "description": "A viral disease causing mottled patterns on apple leaves.",
        "symptoms": "Yellow and green mosaic patterns on leaves. Stunted growth possible.",
        "treatment": "No cure for viral diseases. Remove infected plants to prevent spread.",
        "prevention": "Use virus-free planting material. Control insect vectors."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "This blueberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Healthy green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper soil pH and nutrition."
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "This cherry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care."
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease causing white powdery coating on leaves.",
        "symptoms": "White powdery spots on leaves and shoots. Leaves may curl and distort.",
        "treatment": "Apply sulfur-based or systemic fungicides. Remove heavily infected parts.",
        "prevention": "Ensure good air circulation. Avoid overhead watering. Plant resistant varieties."
    },
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Gray Leaf Spot",
        "description": "A fungal disease caused by Cercospora zeae-maydis.",
        "symptoms": "Rectangular gray to tan lesions on leaves running parallel to veins.",
        "treatment": "Apply foliar fungicides. Use resistant hybrids.",
        "prevention": "Rotate crops. Till under crop residue. Plant resistant varieties."
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "description": "A fungal disease caused by Puccinia sorghi.",
        "symptoms": "Small, circular to elongated brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicides if severe. Usually not economically damaging.",
        "prevention": "Plant resistant hybrids. Early planting can help avoid infection."
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": "This corn leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Normal green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper nutrition and irrigation."
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease caused by Exserohilum turcicum.",
        "symptoms": "Long, elliptical gray-green to tan lesions on leaves.",
        "treatment": "Apply foliar fungicides. Remove crop debris.",
        "prevention": "Use resistant hybrids. Rotate crops. Till under residue."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Guignardia bidwellii.",
        "symptoms": "Brown circular spots on leaves. Fruit shrivels and turns black (mummies).",
        "treatment": "Apply fungicides from bud break. Remove mummified fruit.",
        "prevention": "Prune for good air circulation. Remove infected plant material."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease affecting grapevines.",
        "symptoms": "Tiger-stripe pattern on leaves. Dark spots on berries. Sudden vine collapse.",
        "treatment": "No effective treatment. Remove severely affected vines.",
        "prevention": "Avoid large pruning wounds. Paint pruning cuts with fungicide."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "This grape leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper vineyard management."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis)",
        "description": "A fungal disease causing leaf spots on grapevines.",
        "symptoms": "Irregular brown spots on leaves with dark borders.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "description": "A devastating bacterial disease spread by psyllid insects.",
        "symptoms": "Yellowing of leaves in blotchy pattern. Misshapen, bitter fruit. Tree decline.",
        "treatment": "No cure. Remove infected trees. Control psyllid vectors.",
        "prevention": "Use disease-free nursery stock. Control Asian citrus psyllid."
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas campestris.",
        "symptoms": "Small, dark spots on leaves that may fall out. Fruit has sunken spots.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation."
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "This peach leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular care and monitoring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting pepper plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based sprays. Remove infected plants.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering."
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": "This pepper leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain good growing conditions."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings (target-like) on older leaves.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Rotate crops. Use certified seed. Maintain plant vigor."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "This potato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper irrigation."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by Phytophthora infestans (caused Irish Potato Famine).",
        "symptoms": "Water-soaked spots that turn brown. White mold on leaf undersides. Rapid plant death.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Avoid overhead irrigation. Destroy infected tubers."
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "This raspberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Good pruning and air circulation."
    },
    "Rice___Brown_Spot": {
        "plant": "Rice",
        "disease": "Brown Spot",
        "description": "A fungal disease caused by Bipolaris oryzae.",
        "symptoms": "Oval brown spots on leaves with gray centers.",
        "treatment": "Apply fungicides. Improve soil fertility.",
        "prevention": "Use resistant varieties. Balanced fertilization."
    },
    "Rice___Healthy": {
        "plant": "Rice",
        "disease": "Healthy",
        "description": "This rice leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper water and nutrient management."
    },
    "Rice___Leaf_Blast": {
        "plant": "Rice",
        "disease": "Leaf Blast",
        "description": "A serious fungal disease caused by Magnaporthe oryzae.",
        "symptoms": "Diamond-shaped spots with gray centers and brown borders.",
        "treatment": "Apply systemic fungicides. Drain fields if possible.",
        "prevention": "Use resistant varieties. Avoid excess nitrogen."
    },
    "Rice___Neck_Blast": {
        "plant": "Rice",
        "disease": "Neck Blast",
        "description": "A severe form of rice blast affecting the panicle neck.",
        "symptoms": "Brown to black lesions on panicle neck. Panicle may break and fall.",
        "treatment": "Apply fungicides before heading. Remove infected panicles.",
        "prevention": "Plant resistant varieties. Balanced fertilization."
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "This soybean leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Crop rotation and proper spacing."
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": "A common fungal disease affecting cucurbits.",
        "symptoms": "White powdery patches on leaves. Leaves may yellow and die.",
        "treatment": "Apply fungicides or baking soda solution. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "This strawberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Proper spacing and mulching."
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": "A fungal disease caused by Diplocarpon earlianum.",
        "symptoms": "Irregular purple spots that merge. Leaf margins appear burned.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Use resistant varieties. Renovate beds after harvest."
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting tomato plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings on lower leaves first.",
        "treatment": "Apply fungicides. Remove infected leaves. Mulch around plants.",
        "prevention": "Rotate crops. Stake plants. Water at base of plants."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "This tomato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "A destructive disease caused by Phytophthora infestans.",
        "symptoms": "Large, irregular brown spots. White mold on undersides. Rapid spread.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Improve air circulation. Avoid wet foliage."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease caused by Passalora fulva.",
        "symptoms": "Pale green to yellow spots on upper leaf surface. Olive-brown mold below.",
        "treatment": "Improve ventilation. Apply fungicides. Remove infected leaves.",
        "prevention": "Reduce humidity. Space plants properly. Use resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "A common fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small, circular spots with dark borders and gray centers with black dots.",
        "treatment": "Apply fungicides. Remove infected lower leaves.",
        "prevention": "Rotate crops. Mulch around plants. Avoid overhead watering."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "description": "Tiny arachnid pests that feed on plant cells.",
        "symptoms": "Stippled, yellowing leaves. Fine webbing on undersides. Leaf drop.",
        "treatment": "Spray with water or insecticidal soap. Use miticides if severe.",
        "prevention": "Maintain plant health. Avoid dusty conditions. Introduce predatory mites."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "A fungal disease caused by Corynespora cassiicola.",
        "symptoms": "Brown spots with concentric rings giving target-like appearance.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A highly contagious viral disease.",
        "symptoms": "Mottled light and dark green pattern on leaves. Distorted growth.",
        "treatment": "No cure. Remove and destroy infected plants.",
        "prevention": "Use virus-free seeds. Disinfect tools. Wash hands before handling."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "description": "A devastating viral disease spread by whiteflies.",
        "symptoms": "Upward curling of leaves. Yellowing. Stunted growth. Reduced fruit.",
        "treatment": "No cure. Remove infected plants. Control whiteflies.",
        "prevention": "Use resistant varieties. Control whitefly populations. Use reflective mulches."
    },
    "Wheat brown spot disease": {
        "plant": "Wheat",
        "disease": "Brown Spot",
        "description": "A fungal disease affecting wheat leaves.",
        "symptoms": "Brown oval spots on leaves that may merge.",
        "treatment": "Apply fungicides. Remove crop residue.",
        "prevention": "Use resistant varieties. Crop rotation."
    }
}


def load_model_and_classes():
    """Load the model and class indices"""
    global model, class_indices
    
    print("🔄 Loading AI model...")
    model = load_model('models/99pct_final_reached.h5')
    
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Reverse the indices to get class names from predictions
    class_indices = {v: k for k, v in class_indices.items()}
    print("✅ Model loaded successfully!")


def predict_disease(img_path):
    """Predict disease from image"""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100
    
    # Get class name
    class_name = class_indices.get(predicted_class_idx, "Unknown")
    
    # Get disease info
    info = DISEASE_INFO.get(class_name, {
        "plant": class_name.split("___")[0] if "___" in class_name else "Unknown",
        "disease": class_name.split("___")[1] if "___" in class_name else class_name,
        "description": "Information not available for this disease.",
        "symptoms": "Please consult a plant pathologist for detailed symptoms.",
        "treatment": "Consult with local agricultural extension for treatment options.",
        "prevention": "Maintain good plant hygiene and regular monitoring."
    })
    
    return {
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "plant": info["plant"],
        "disease": info["disease"],
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "is_healthy": "healthy" in class_name.lower()
    }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            result = predict_disease(filepath)
            
            # Read image for preview
            with open(filepath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            result['image'] = f"data:image/jpeg;base64,{img_data}"
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file'}), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    load_model_and_classes()
    print("\n🌿 Leaf Disease Detection Web App")
    print("=" * 40)
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=False)
