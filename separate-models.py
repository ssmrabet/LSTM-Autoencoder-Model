from keras import layers, Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector

class Autoencoder():
    def __init__(self,verbose=1,epochs=50,batch_size=16,loss='mse',optimizer='adam'):
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer

    def create_model(self, timesteps, features):
        inputs = Input(shape=(timesteps, features),name='inputs')
        # encoder model
        encoder = Sequential()
        encoder.add(inputs)

        # for variable input data length add 0.0 to columns and lines where the seequence length less then others
        # then mask the 0.0 with this Masking line
        encoder.add(Masking(mask_value=0.0))
        encoder.add(LSTM(int(timesteps/2), activation='relu', return_sequences=True))
        encoder.add(LSTM(int(timesteps/3), activation='relu', return_sequences=True))
        encoder.add(LSTM(int(timesteps/4), activation='relu', return_sequences=True))
        encoder.add(LSTM(10, activation='relu', return_sequences=False))
        
        # decoder model
        decoder = Sequential()
        decoder.add(RepeatVector(timesteps))
        decoder.add(LSTM(int(timesteps/4), activation='relu', return_sequences=True))
        decoder.add(LSTM(int(timesteps/3), activation='relu', return_sequences=True))
        decoder.add(LSTM(int(timesteps/2), activation='relu', return_sequences=True))
        decoder.add(LSTM(features, activation='relu', return_sequences=True))
        
        # autoencoder model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder_model')

        # compile model
        autoencoder.compile(loss=self.loss, optimizer=self.optimizer)

        # summary
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

        return autoencoder, encoder, decoder
        
    def train_model(self, model, X_train):
        # fit network
        model.fit(X_train, X_train, 
                    epochs=self.epochs, 
                    batch_size=self.batch_size, 
                    verbose=self.verbose)
