import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten,Dropout
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint,CSVLogger

token = '/home/moiz/Downloads/sentiment/Flickr8k.token.txt'

captions = open(token, 'r').read().strip().split('\n')

d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]

d['2641268201_693b08cb0e.jpg']

images = '/home/moiz/Downloads/sentiment/Flicker8k_Images/'

img = glob.glob(images+'*.jpg')

img[:5]

train_images_file = '/home/moiz/Downloads/sentiment/Flickr_8k.trainImages.txt'

train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp

train_img = split_data(train_images)
len(train_img)

val_images_file = '/home/moiz/Downloads/sentiment/Flickr_8k.valImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

val_img = split_data(val_images)
len(val_img)

test_images_file = '/home/moiz/Downloads/sentiment/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

test_img = split_data(test_images)
len(test_img)

Image.open(train_img[0])

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

plt.imshow(np.squeeze(preprocess(train_img[0])))

model = InceptionV3(weights='imagenet')

from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)

tryi = model_new.predict(preprocess(train_img[0]))

tryi.shape

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

encoding_train = {}
for img in tqdm(train_img):
    encoding_train[img[len(images):]] = encode(img)

with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))

encoding_train['2385662157_d09d42bef7.jpg'].shape

encoding_test = {}
for img in tqdm(test_img):
    encoding_test[img[len(images):]] = encode(img)

 with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))

encoding_test[test_img[0][len(images):]].shape

train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]

len(train_d)

val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]

len(val_d)

test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]

len(test_d)

caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')

words = [i.split() for i in caps]

unique = []
for i in words:
    unique.extend(i)

unique = list(set(unique))

unique = pickle.load(open('/home/moiz/Downloads/sentiment/unique.p', 'rb'))

len(unique)

word2idx = {val:index for index, val in enumerate(unique)}

word2idx['<start>']

idx2word = {index:val for index, val in enumerate(unique)}

idx2word[5553]

max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)
max_len

len(unique), max_len

vocab_size = len(unique)

vocab_size

f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

for key, val in train_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()

df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')

len(df)

c = [i for i in df['captions']]
len(c)

imgs = [i for i in df['image_id']]

a = c[-1]
a, imgs[-1]

for i in a.split():
    print (i, "=>", word2idx[i])

samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1

samples_per_epoch

steps = len(train_img)

def data_generator(batch_size):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])

        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    
                    n = np.zeros(vocab_size)
                   
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0

embedding_size = 300

image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        Dropout(0.5),
        RepeatVector(max_len)
    ])
image_model.summary()

caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len,mask_zero = True),
        Dropout(0.5),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])
caption_model.summary()

final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.summary()

final_model.load_weights('/home/moiz/Downloads/sentiment/w1.h5')

checkpointer = ModelCheckpoint(filepath= "checkpoint.hdf5",save_best_only = True)
csvloggr = CSVLogger("history.log",'w')
history = final_model.fit_generator(data_generator(batch_size=512), samples_per_epoch=samples_per_epoch, nb_epoch=100)
model.save(final_model)

final_model.save_weights('/home/moiz/Downloads/sentiment/w1.h5')

def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

def beam_search_predictions(image, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
           
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

try_image = test_img[508]
Image.open(try_image)

print ('Normal Max search:', predict_captions(try_image)) 
print ('Beam Search, k=3:', beam_search_predictions(try_image, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image, beam_index=7))

t = try_image[-25:]
p = {}
p = d[t]
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
reference = p[0]
candidate = beam_search_predictions(try_image, beam_index=3)
cc = SmoothingFunction()
score = sentence_bleu(reference, candidate, smoothing_function=cc.method4)
print(score)
print(p[0])
print(candidate)
if score < 0.3:
    print("bad prediction")
else:
    print("good prediction")

import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
npix = 299
target_size = (npix,npix,3)
fig = plt.figure(figsize=(100,800))
npic = len(reference)
filename = try_image
image_load = load_img(filename, target_size=target_size)
ax = fig.add_subplot(npic,2,1,xticks=[],yticks=[])
ax.imshow(image_load)
ax = fig.add_subplot(npic,2,2)
plt.axis('off')
ax.plot()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.text(0,0.7,"true:" + reference,fontsize=60)
ax.text(0,0.4,"pred:" + beam_search_predictions(try_image, beam_index=3),fontsize=60)
ax.text(0,0.1,"BLEU: {}".format(score),fontsize=60)
plt.show()

import pickle
file_name = "/home/moiz/Downloads/sentiment/psychexp.txt" 

def read_file(file_name): 
    data_list  = []
    with open(file_name, 'r') as f: 
        for line in f: 
            line = line.strip() 
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data_list.append([label, text])
    return data_list


psychExp_txt = read_file(file_name)

print("The number of instances: {}".format(len(psychExp_txt)))

print("Data example: ")
print(psychExp_txt[0])

import re 
from collections import Counter

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 

    # 1. treat alphanumeric characters as word tokens
    # Since tweets contain #, we keep it as a feature
    # Then, extract all ngram lengths
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)
    
    # 2. treat punctuations as word token
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    
    # 3. Return a dictinaory whose keys are the list of elements 
    # and their values are the number of times appearede in the list.
    return Counter(text_features)

print(create_feature("I miss you!"))
print(create_feature(" aly wins the silver!!!!!!  #olympics"))
print(create_feature(" aly wins the gold!!!!!!  #olympics", (1, 2)))

def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in psychExp_txt:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

print("features example: ")
print(X_all[0])

print("Label example:")
print(y_all[0])

import sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)

from sklearn.metrics import accuracy_score

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Classifiers 
svc = SVC()
lsvc = LinearSVC(random_state=123)
rforest = RandomForestClassifier(random_state=123)
dtree = DecisionTreeClassifier()

clifs = [svc, lsvc, rforest, dtree]

# train and test them 
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
for clf in clifs: 
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))


from sklearn.model_selection import GridSearchCV

parameters = {'C':[1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100], 
             'tol':[0.1, 0.01, 0.001, 0.0001, 0.00001]}

lsvc = LinearSVC(random_state=123)
grid_obj = GridSearchCV(lsvc, param_grid = parameters, cv=5)
grid_obj.fit(X_train, y_train)

print("Validation acc: {}".format(grid_obj.best_score_))
print("Training acc: {}".format(accuracy_score(y_train, grid_obj.predict(X_train))))
print("Test acc    : {}".format(accuracy_score(y_test, grid_obj.predict(X_test))))
print("Best parameter: {}".format(grid_obj.best_params_))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, grid_obj.predict(X_test))
print(matrix)

matrix = [[112 , 27  ,16  ,23 , 12,  19,  19],
         [ 20 ,113  ,11 , 11  ,12  ,17   ,9],
         [  8  ,19 ,157  , 3  ,11  ,16  ,10],
         [ 19  ,12  ,15 ,107  ,12  ,17  ,36],
         [  7  , 9   ,7  , 7 ,155  ,18   ,8],
         [ 17   ,9 , 17  ,12  ,21 ,137  ,10],
         [ 31  ,19  ,10  ,30 , 17  ,10  ,82]]

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
df_cm = pd.DataFrame(matrix, index = l, columns = l)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt="d",cmap="Set1")
plt.show()

label_freq = {}
for label, _ in psychExp_txt: 
    label_freq[label] = label_freq.get(label, 0) + 1

# print the labels and their counts in sorted order 
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"🤢", "shame":"😅", "guilt":"😔"}

t1 = predict_captions(try_image)
t2 = beam_search_predictions(try_image, beam_index=3)
t3 = beam_search_predictions(try_image, beam_index=5)
t4 = beam_search_predictions(try_image, beam_index=7)
texts = [t1, t2, t3, t4]
for text in texts: 
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = grid_obj.predict(features)[0]
    print("{} {}".format(text,emoji_dict[prediction]))

