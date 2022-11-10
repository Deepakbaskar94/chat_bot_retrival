import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, activation, Dropout
from keras.optimizers import SGD
# nltk.download('omw-1.4')

lanmatizer = WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore_words=['?','!','.']

##########################################################################################################################
#reading the file
##########################################################################################################################

data_file = open('intents.json').read()
intents = json.loads(data_file)


##########################################################################################################################
#tokenize words, classes, documents
##########################################################################################################################
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # print(pattern)
        w=nltk.word_tokenize(pattern)
        # print(w)
        # print('token is: ', w)
        words.extend(w)
        # print('words: ', words)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print('Documents: \n',documents)
# print('words: \n', words)
# print('classes: \n', classes)

##########################################################################################################################
#lemmatize
##########################################################################################################################

words = [lanmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))
# print('new words: \n', words)
# print('classes:\n', classes)

#writing to binary file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


##########################################################################################################################
#training bag for input and output
##########################################################################################################################
training = []
output_empty = [0]*len(classes)
# [0,0,0,0,0,0,0]

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lanmatizer.lemmatize(word.lower()) for word in pattern_words]
    # print('current pattern words: \n', pattern_words)

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # print('current bag: \n', bag)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    # print('current output: \n', output_row)

    training.append([bag, output_row])
print('training: \n',training)

random.shuffle(training)
training = np.array(training,dtype=object,)

train_x = list(training[:,0])
train_y = list(training[:,1])
# print('X:',train_x)
# print('Y:',train_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#compilling the model & define an optimizer function
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

mfit = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', mfit)

# print('Created my first model')
