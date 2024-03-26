"""import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import pickle
import json

with open('intents1.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)  # saving these variables in the file
except:
    # these blank lists are created as we want to go through the json file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # tokenize the words, stemming. Root words are formed
            wrds = nltk.word_tokenize(pattern)  # returns a list of tokenized words
            words.extend(wrds)  # adds to words
            docs_x.append(wrds)
            # gives what intent the tag is a part of
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # removing all the duplicate elements
    words = [stemmer.stem(w.lower()) for w in words if
             w != "?"]  # removing any question marks to not have any meaning to our model, and stemming
    words = sorted(list(set(words)))  # set removes the duplicate elements

    labels = sorted(labels)  # sorting the labels

    # create training and testing output
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    # neural network does not understand strings, but only numbers.
    # presenting the words into numbers
    for x, doc in enumerate(docs_x):
        # list to keep a check on what words are present
        # stemming the words
        bag = []  # bag of words
        wrds = [stemmer.stem(w.lower()) for w in doc]
        # going through the words and adding the information to bag
        for w in words:
            if w in wrds:  # word exsits so add 1 to the list
                bag.append(1)
            else:  # word does not exsit so add 0 to the list
                bag.append(0)

        output_row = out_empty[:]
        # where the tag is in our labels, and set value to 1 in output
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # turning the lists into nparrays to be able to fed into model
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# resetting the graph data, to get rid of previous settings
tensorflow.compat.v1.reset_default_graph()
# defines the input shape for our model
net = tflearn.input_data(shape=[None, len(training[0])])
# 8 neurons for the first hidden layer
net = tflearn.fully_connected(net, 8)
# 8 neurons for the second hidden layer
net = tflearn.fully_connected(net, 8)
# gets probability for each neuron in the output layer,
# the neuron which has the highest probability is selected
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  #
net = tflearn.regression(net)
#print(model.summary())
# training the model
model = tflearn.DNN(net)  # deep neural network
#print(model.summary())
try:
    x
    model.load("model.tflearn")
except:
    # we show the model the data 1000 times, the more times it sees the data, the more accurate it should get
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)  # list of tokenized words
    s_words = [stemmer.stem(word.lower()) for word in s_words]  # stemming the words

    for x in s_words:
        for i, w in enumerate(words):
            if w == x:  # if current word is equal to our word in the sentence, then add 1 to bag list, generates the bag of words
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("You can start talking to Bubble! If you wish to end the conversation, please type 'quit'")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":  # way to get out of the program
            break

        results = model.predict(
            [bag_of_words(inp, words)])  # makes prediction, this only gives us some probability, no meaningful output
        results_index = numpy.argmax(
            results)  # this gives the index of the tag with the greatest probability in our list
        tag = labels[results_index]  # maps the word to a particular tag
        # if results[results_index] > 0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))  # selects a response from the tag


chat()
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import pickle
import json
import speech_recognition as sr
import pyttsx3

# Initialize NLP components
stemmer = LancasterStemmer()
nltk.download('punk')

# Load intents data
with open('intents1.json') as file:
    data = json.load(file)

# Load or create pickle file
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Reset TensorFlow graph
tensorflow.compat.v1.reset_default_graph()

# Define neural network architecture
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Initialize and train the model
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Initialize Speech Recognition and Text-to-Speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()


# Function to convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("You:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return ""


# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


# Function to process user input and generate response
def process_input(input_text):
    global responses
    results = model.predict([bag_of_words(input_text, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    response = random.choice(responses)
    return response

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for x in s_words:
        for i, w in enumerate(words):
            if w == x:
                bag[i] = 1

    return numpy.array(bag)


# Main chat function
def chat():
    text_to_speech("You can start talking to Bubble! If you wish to end the conversation, please say 'quit'")
    while True:
        input_text = speech_to_text()
        if input_text.lower() == "quit":
            break
        response = process_input(input_text)
        text_to_speech(response)


# Run the chat
chat()


