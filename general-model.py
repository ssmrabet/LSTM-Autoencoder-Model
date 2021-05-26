from keras import layers, Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, TimeDistributed

class Autoencoder():
    def __init__(self,verbose=1,epochs=50,batch_size=16,loss='mse',optimizer='adam'):
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer

    def create_model(self, timesteps, features):
        
        autoencoder = Sequential()

        autoencoder.add(Masking(mask_value=0.0, input_shape=(timesteps, features)))
        # layer 1, masking = (None, x, y)
        autoencoder.add(LSTM(timesteps, activation='relu', return_sequences=True))
        # layer 2, lstm = (None, x, x)
        autoencoder.add(LSTM(int(timesteps/2), activation='relu', return_sequences=False))
        # layer 3, lstm_1 = (None, x/2)

        autoencoder.add(RepeatVector(timesteps))
        # The RepeatVector layer acts as a bridge between the encoder and decoder modules
        # RepeatVector(x), replicates the feature vector x times.
        # layer 4, repeat vector = (None, x, x/2)

        autoencoder.add(LSTM(int(timesteps/2), activation='relu', return_sequences=True))
        # layer 5, lstm_2 = (None, x, x/2)
        autoencoder.add(LSTM(timesteps, activation='relu', return_sequences=True))
        # layer 6, lstm_3 = (None, x, x)
        autoencoder.add(TimeDistributed(Dense(features, activation='relu')))
        # This wrapper allows to apply a layer to every temporal slice of an input
        # TimeDistributed(Dense(n)), is added in the end to get the output, 
        # where n is the number of features in the input data
        # layer 5, time distributed = (None, x, y)

        # compile model
        autoencoder.compile(loss='mse', optimizer='adam')

        # summary
        autoencoder.summary()

        return autoencoder

    def train_model(self, model, X_train):
        # fit network
        model.fit(X_train, X_train, 
                    epochs=self.epochs, 
                    batch_size=self.batch_size, 
                    verbose=self.verbose)