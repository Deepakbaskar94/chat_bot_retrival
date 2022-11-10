import tkinter
from tkinter import *

import pickle
import json
import random

from keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np

lanmatizer = WordNetLemmatizer()


model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())


def bow(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lanmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1

    return (np.array(bag))

def predict_class(sentence):
    sentence_bag = bow(sentence)
    res = model.predict(np.array([sentence_bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probablity': str(r[1])})

    return return_list


def getresponse(ints):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result= random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg)
    res = getresponse(ints)
    return res


def send():
    msg = TextEntryBox.get("1.0", "end-1c").strip()
    TextEntryBox.delete('1.0', 'end')

    if msg != '':
        chathistory.config(state=NORMAL)
        chathistory.insert('end', 'you: ' + msg + "\n")

        res = chatbot_response(msg)
        chathistory.insert('end', 'Bot: ' + res + '\n')
        chathistory.config(state=DISABLED)
        chathistory.yview('end')

    # pass

base = Tk()
base.title("ProbePlus")
base.geometry("412x612")
base.resizable(width=False, height=False)

#chat history textview
chathistory = Text(base, bd=0, bg='white', font='Arial')
chathistory.config(state=DISABLED)

sendbutton = Button(base, font=('Arial', 12, 'bold'), text='Send', bg="#dfdfdf", activebackground="#84f5a2", fg="#000000", command=send)
# sendbutton = Button(base, font=('Arial', 12, 'bold'), text='Send', bg="red", command=send)
TextEntryBox = Text(base, bd=0, bg='white', font='Arial')

chathistory.place(x=6, y=6, height=550, width=400)
sendbutton.place(x=306, y=556, height=50, width=100)
TextEntryBox.place(x=6, y=556, height=50, width=300)


base.mainloop()