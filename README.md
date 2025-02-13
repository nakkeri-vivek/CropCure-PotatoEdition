# Potato_Disease_Detection
This project utilizes machine learning and computer vision techniques to detect diseases in potato leaves. The goal is to build a system that can help farmers identify early signs of disease and take necessary actions to prevent crop loss.

Table of Contents
Description
Technologies Used
Installation
Usage
Dataset
Model
Contributing
License
Description
Potato crops are highly vulnerable to various diseases that can impact yield and quality. This project aims to provide a solution by detecting diseases such as late blight, early blight, and other fungal or bacterial infections using images of potato leaves. The system uses a convolutional neural network (CNN) to classify leaf images into healthy or diseased categories.

Technologies Used
Python
OpenCV
TensorFlow/Keras
NumPy
Matplotlib
scikit-learn
Installation
Clone this repository to your local machine:

bash
Copy
Edit
git clone https://github.com/your-username/potato-leaf-disease-detection.git

Navigate to the project directory:
cd potato-leaf-disease-detection

Install the required dependencies:
pip install -r requirements.txt
Usage
Run the main script to start the detection process:

bash
Copy
Edit
python detect_disease.py
You can input images of potato leaves for detection and the program will classify them as either healthy or diseased.

To train the model with a custom dataset:

bash
Copy
Edit
python train_model.py
This will train the model on the provided dataset and save the trained model for later use.

Dataset
The dataset used for this project is the Potato Leaf Disease Dataset, which consists of labeled images of healthy and diseased potato leaves. The dataset can be found at Kaggle's Potato Disease Dataset.

Model
The project utilizes a Convolutional Neural Network (CNN) to classify images of potato leaves. The model is trained on labeled images and can detect common diseases like late blight and early blight. For detailed information on model architecture, refer to the train_model.py script.

Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests. Please make sure to follow the coding standards and guidelines provided in the repository.
