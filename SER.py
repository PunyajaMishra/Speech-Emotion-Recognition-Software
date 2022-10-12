import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        #we call our soundfile 'X'
        X = sound_file.read(dtype="float32")

        #get the sample rate of our audio file
        sample_rate=sound_file.samplerate

        #if chroma is true, then get the short time fourier transform of X
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])

        #for each feature of the three, 
        # if it exists, make a call to the corresponding function from librosa.feature (eg- librosa.feature.mfcc for mfcc)
        # get the mean value
        #store this in the result . hstack() stacks arrays in sequence horizontally
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))

        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))

        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# OUR DICTIONARY
#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
#info for training our data
def load_data(test_size=0.2):
    x,y=[],[]
    #for every audio file
    for file in glob.glob("ravdess data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        #function checks whether this emotion is in our list of observed_emotions
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue #to next file
        
        #makes a call to extract_feature and stores what is returned in ‘feature’
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        #store our feature and emotion
        x.append(feature)
        y.append(emotion)
    
    #calling our training data
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#DataFlair - Split the dataset
#split the dataset into training and testing sets
#test set 25%
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#DataFlair - Get the shape of the training and testing datasets
#Observe the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
#This is a Multi-layer Perceptron Classifier; 
# it optimizes the log-loss function using LBFGS or stochastic gradient descent. 
# Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network 
# for the purpose of classification. This is a feedforward ANN model.
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))