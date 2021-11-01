import numpy as np
from tensorflow import keras

def predictPlate(letters):
    output_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    model1 = keras.models.load_model('models/mnist_32_30.h5')
    model2 = keras.models.load_model('models/emnist_32_30.h5')

    final_result = ''

    for symbol in [i[1] for i  in letters]:
        inverted1 = 1 - symbol / 255

        inverted2 = np.rot90(inverted1, 3)
        inverted2 = np.fliplr(inverted2)


        inverted1 = np.expand_dims(inverted1, -1)
        inverted1 = inverted1.reshape((1, 28, 28, 1))
        inverted2 = np.expand_dims(inverted2, -1)
        inverted2 = inverted2.reshape((1, 28, 28, 1))

        vector1 = model1.predict(inverted1)[0]
        vector2 = model2.predict(inverted2)[0]

        prediction1 = max(vector1)
        prediction2 = max(vector2)

        value1 = output_values[vector1.tolist().index(prediction1)]
        value2 = output_values[vector2.tolist().index(prediction2)]

        final_result += value1 if prediction1 > 0.997 and prediction2 < 0.6 else value2 

    return final_result
