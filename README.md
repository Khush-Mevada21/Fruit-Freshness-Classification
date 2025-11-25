# Fruit Freshness Classification 

- This project automates the process of freshness detection in fruits for warehouse conveyor-belt systems.Using high-speed cameras and a ResNet50-based deep learning model, the system classifies fruit crates into:<br>- Fresh <br> - Spoiled

## Tech Stack

### **🔹 Machine Learning & Deep Learning**
- **PyTorch** — Model training, inference, and GPU acceleration  
- **Torchvision** — Image datasets, transforms, ResNet50 architecture  
- **ResNet50 (Transfer Learning)** — Pre-trained on ImageNet for feature extraction  

---

### **🔹 Data Processing**
- **Pillow (PIL)** — Image loading and manipulation  
- **Torchvision Transforms** — Resize, normalization, tensor conversion  
- **Pickle** — Storing preprocessing pipelines  
- **JSON** — Saving class-to-index mapping  

---

### **🔹 Model Optimization**
- **Adam Optimizer**  
- **CrossEntropyLoss**  
- **StepLR Learning Rate Scheduler**  
- **Accuracy Metrics (Train/Validation/Test)**  

---

### **🔹 Deployment & UI**
- **Streamlit** — Drag-and-drop web app for image prediction  
- **Python 3.10+** — Main environment  


