MediScan AI: Lung Disease Prediction System
📝 DescriptionMediScan AI is an automated, deep learning-based Clinical Decision Support System (CDSS) designed to assist medical professionals in the detection and classification of 14 distinct thoracic conditions from Chest X-rays. The project bridges the gap between complex Deep Learning architectures and clinical usability through an interactive web-based interface.  

🚀 Key FeaturesDeep Learning Engine: Utilizes the EfficientNetB0 Convolutional Neural Network (CNN) architecture, optimized for high-performance image classification.  
Robust Preprocessing: Includes a custom data pipeline to handle and normalize 16-bit PNG medical imagery, ensuring numerical stability and preventing model instability.  
Explainable AI (XAI): Integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual heatmaps, providing clinicians with transparency regarding the model's focus regions.  
Interactive UI: Built with Flask and JavaScript to allow medical staff to upload X-ray images and receive real-time disease probabilities and interpretability heatmaps.  

🛠 Technologies UsedPython: Core logic and backend development.  
TensorFlow/Keras: Implementation of the EfficientNetB0 architecture.  
Flask: Backend web framework for handling image processing and API requests.  
HTML/CSS/JavaScript: Frontend interface for user interaction and dynamic heatmap rendering.  

⚙️ How to RunClone this repository: git clone [https://github.com/MoizPatel/lung-disease-prediction](https://github.com/MoizPatel/MediScan-AI-Lung-Disease-Prediction-System)
Install the required libraries: pip install -r requirements.txt
Launch the application: python src/app.py
Access the interface: Open your browser and navigate to http://127.0.0.1:5000/
