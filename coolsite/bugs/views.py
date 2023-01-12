#************************* pip install scikit-learn

import pickle

import numpy as np
import tensorflow
from django.contrib import admin
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from keras.models import load_model

admin.autodiscover()

# Create your views here.

menu = [{"name": "CNN", "url": "p_CNN"},
        {"name": "RNN", "url": "p_RNN"},
        {"name": "TransferLearning", "url": "p_Transfer"}]


species=['Муравей', 'Пчела', 'Бабочка', 'Гусеница', 'Богомол', 'Паук', 'Рогач']

def index(request):
    return render(request, 'index.html', {'title': 'Главная страница', 'menu': menu})


def f_CNN(request):
    new_model = load_model('NeuroModels/bugs_0_CNNModel.h5')
    if request.method == 'POST' and request.FILES:
        # получаем загруженный файл
        file = request.FILES['myfile1']
        fs = FileSystemStorage()
        # сохраняем на файловой системе
        filename = fs.save(file.name, file)
        # получение адреса по которому лежит файл
        file_url = fs.url(filename)
        img = tensorflow.keras.utils.load_img(
            fs.path(filename), target_size=(180, 180)
        )
        img_array = tensorflow.keras.utils.img_to_array(img)
        img_array = tensorflow.expand_dims(img_array, 0)  # Create a batch

        predictions = new_model.predict(img_array)
        score = tensorflow.nn.softmax(predictions[0])
        return render(request, 'CNN.html', {'title': 'CNN модель',
                                               'menu': menu,
                                               'PredictedSpecie': species[
                                                   np.argmax(score)],
                                               'file_url':file_url
                                               })
    return render(request, 'CNN.html',
                  {'title': 'CNN модель', 'menu': menu})


def f_RNN(request):
    new_model = load_model('NeuroModels/bugs_1_RnnModel.h5')
    if request.method == 'POST' and request.FILES:
        # получаем загруженный файл
        file = request.FILES['myfile1']
        fs = FileSystemStorage()
        # сохраняем на файловой системе
        filename = fs.save(file.name, file)
        # получение адреса по которому лежит файл
        file_url = fs.url(filename)
        img = tensorflow.keras.utils.load_img(
            fs.path(filename), target_size=(180, 180)
        )
        img_array = tensorflow.keras.utils.img_to_array(img)
        img_array = tensorflow.expand_dims(img_array, 0)  # Create a batch

        predictions = new_model.predict(img_array)
        score = tensorflow.nn.softmax(predictions[0])
        return render(request, 'RNN.html', {'title': 'RNN модель',
                                               'menu': menu,
                                               'PredictedSpecie': species[
                                                   np.argmax(score)],
                                               'file_url':file_url
                                               })
    return render(request, 'RNN.html',
                  {'title': 'RNN модель', 'menu': menu})

def f_Transf(request):
    new_model = load_model('NeuroModels/bugs_2_TransModel.h5')
    if request.method == 'POST' and request.FILES:
        # получаем загруженный файл
        file = request.FILES['myfile1']
        fs = FileSystemStorage()
        # сохраняем на файловой системе
        filename = fs.save(file.name, file)
        # получение адреса по которому лежит файл
        file_url = fs.url(filename)
        img = tensorflow.keras.utils.load_img(
            fs.path(filename), target_size=(160, 160)
        )
        img_array = tensorflow.keras.utils.img_to_array(img)
        img_array = tensorflow.expand_dims(img_array, 0)  # Create a batch

        predictions = new_model.predict(img_array)
        score = tensorflow.nn.softmax(predictions[0])
        return render(request, 'Transf.html', {'title': 'TransferLearning модель',
                                               'menu': menu,
                                               'PredictedSpecie':species[np.argmax(score)],
                                               'file_url':file_url
        })
    return render(request, 'Transf.html', {'title': 'TransferLearning модель', 'menu': menu})
