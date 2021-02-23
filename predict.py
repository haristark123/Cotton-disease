#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('Cotton.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image.reshape(1, 64, 64, 3)
        result = np.argmax(model.predict(test_image),axis=-1)

        # if result[0][0] == 1:
        #     prediction = 'dog'
        #     return [{ "image" : prediction}]
        if result == 0:
            prediction = 'diseased Leaf'
            return [{"image": prediction}]
        elif result == 1:
            prediction = 'diseased plant'
            return [{"image": prediction}]
        elif result == 2:
            prediction = 'fresh leaf'
            return [{"image": prediction}]
        else:
            prediction = 'fresh plant'
            return [{"image": prediction}]
        # else:
        #     prediction = 'cat'
        #     return [{ "image" : prediction}]


