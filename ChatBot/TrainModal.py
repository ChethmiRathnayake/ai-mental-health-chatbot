import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# ------------------------
# FIX: use legacy optimizer
from tensorflow.keras.optimizers.legacy import SGD
# ------------------------

# WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# corpus pre-processing
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("stressdataset.json").read()
intents = json.loads(data_file)

# tokenization
nltk.download('punkt')
nltk.download('wordnet')

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Training set creation
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = output_empty.copy()
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle & convert
random.shuffle(training)

X = np.array([sample[0] for sample in training])
Y = np.array([sample[1] for sample in training])

# Pad sequences
X = pad_sequences(X, maxlen=len(words), padding='post')

# Neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y[0]), activation='softmax'))

# -------------------------------
# FIX: Legacy SGD supports decay
# -------------------------------
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(X, Y, epochs=100, batch_size=5, verbose=1)

# Save model
model.save('chatbot.h5', hist)

print("\n" + "*" * 50)
print("Model Created Successfully!")
