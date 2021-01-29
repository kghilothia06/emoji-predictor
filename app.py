from flask import Flask , render_template , redirect , request
import pandas as pd
import numpy as np
import emoji

from tensorflow import keras


model = keras.models.load_model('model.pkl')

emoji_dictionary = {
                    "0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":grinning_face_with_big_eyes:",
                    "3": ":disappointed_face:",
                    "4": ":fork_and_knife:"
                    }

f = open('glove.6B.50d.txt' , encoding='utf8')

#make our own word embedding dictionary
embeddings_idx={}

for line in f:
    values = line.split()
    #print(values[0])
    #print(values[1:])
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float')
    #print(word,coefs)
    embeddings_idx[word] = coefs

f.close()


def embedding_output(X):
    
    maxlen=10
    #embedding dimension
    emb_dim = 50
    #batch size
    batch_size = X.shape[0]
    #output of the function
    embedding_out = np.zeros((batch_size , maxlen , emb_dim))
    
    for ix in range(X.shape[0]):
        #tokenize current sentence
        X[ix] = X[ix].split()
        
        for ij in range(len(X[ix])):
            #get current word's embedding from 'glove' embeddings iff that word is present 
            if X[ix][ij].lower() in embeddings_idx.keys() and ij < maxlen:
                embedding_out[ix][ij] = embeddings_idx[X[ix][ij].lower()]
    
    return embedding_out
    

app=Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/home')
def home():
    return redirect('/')

@app.route('/predict',methods=['POST'])
def submit_data():
    if request.method=='POST':
        text = request.form['sentence']
        
        text_series = pd.Series(text)
        emb_out = embedding_output(text_series)
        pred = model.predict_classes(emb_out)
        
        return render_template("index.html" , prediction = 'predicted emoji is {}'.format(emoji.emojize(emoji_dictionary[str(pred[0])])))
    
    return "nothing"
    

if __name__=='__main__':
	app.run()
    
    
    
