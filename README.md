# LSTM-Autoencoder-Model
An LSTM Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture.

![Image of LSTM Autoencoder](https://user-images.githubusercontent.com/34431729/58381821-3ab67400-7ffd-11e9-9f06-6d1045c5ecbc.png)

## LSTM Autoencoder VS Regular Autoencoder
Both LSTM autoencoders and regular autoencoders, encode the input to a compact value, which can then be decoded to reconstruct the original input. While LSTM autoencoders are capable of dealing with sequence as input, regular autoencoders won’t. For example, regular autoencoders will fail to generate a sample sequence for a given input distribution in generative mode whereas LSTM counterpart can. In addition, LSTM can obviously take variable length inputs while regular ones take only fixed size inputs.

example of variable length:
```bash
raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]
```
```python
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)
print(padded_inputs)
```
```bash
[[ 711  632   71    0    0    0]
 [  73    8 3215   55  927    0]
 [  83   91    1  645 1253  927]]
````
for more details you can check [this Masking and padding with Keras link](https://www.tensorflow.org/guide/keras/masking_and_padding)

## General model
is a one general LSTM Autoencoder model

## Separate models
is a 3 models creations

- Encoder

- Decoder

- Autoencoder
