# tflite-mnist-recognition

Runs inference with a bare metal CNN using tflite micro.
Uses a 'distributed convolution' kernel which splits the conv. operation into subsets and a portion of the operations are executed with bit-sliced multiplication scheme.

-Use "digit_recognition_convolutional_model_training.py" or "digit_recognition_convolutional_model_training.ipynb" to train the network and convert the TFLite model into C-arrays.
-A trained model is already provided in "conv_model_array_int8.h"
-"get_image.h" contains many example digits to test the model with.
