# 🧠 Emotion Classification using TensorFlow & Keras

This project classifies text data into different emotions using deep learning (TensorFlow + Keras).  
It tokenizes and encodes text, builds a neural network, and predicts emotions for user-inputted sentences.

---

## 📂 Project Structure

Emotion Classification/
│── index.py # Main script (model training + prediction)
│── train.txt # Dataset for training
│── val.txt # Optional validation dataset
│── text.txt # Example or sample text file
│── .gitignore

---

## ⚙️ Requirements

Install the required dependencies before running the project:

```bash
pip install tensorflow pandas numpy scikit-learn


🚀 How to Run the Project

1. Clone this repository:

git clone https://github.com/YOUR_USERNAME/Emotion-Classification.git
cd Emotion-Classification


2. Make sure your dataset (train.txt) is in the same directory.
The file should have two columns separated by ;:

I am happy;joy
I am sad;sadness
I feel angry;anger


3. Run the script:

python index.py


4. The model will train for 10 epochs and then start an interactive emotion detector:

🎯 Emotion Classifier is ready! Type your text below.
👉 Enter a sentence: I am feeling great today!
💬 Emotion detected: joy


5. Type exit to stop the program.

---------------------------------------

🧩 Model Architecture
. Embedding layer: Converts words into dense vector representations
. Flatten layer: Converts embeddings to 1D format
. Dense layers: Classifies emotions using ReLU and Softmax activations

📊 Example Output
Input Sentence	Predicted Emotion
I am feeling great today!	joy
This is terrible news.	sadness
I am so excited!	surprise


🧠 Future Improvements
. Add pretrained embeddings (like GloVe or Word2Vec)
. Use LSTM or GRU instead of Flatten for better accuracy
. Build a Streamlit or Flask web app for live emotion prediction

👨‍💻 Author
Anshul Dhoundiyal
📧 anshuldhoundiyal05@gmail.com