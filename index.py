import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]
print(data.head())

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encode and one-hot labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(labels)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(
    padded_sequences, one_hot_labels, test_size=0.2
)

# Build model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

# Save model after training (optional)
model.save("emotion_model.h5")

# Interactive user input
print("\n🎯 Emotion Classifier is ready! Type your text below.")
print("Type 'exit' to stop.\n")

while True:
    input_text = input("👉 Enter a sentence: ")
    if input_text.lower() == "exit":
        print("👋 Exiting Emotion Detector. Goodbye!")
        break

    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    print(f"💬 Emotion detected: {predicted_label[0]}\n")
