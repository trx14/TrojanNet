import keras
from itertools import combinations
import math
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Lambda, Add, Activation, Input, Reshape, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
import cv2
import os
import keras.backend as K
import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Backnet:
    def __init__(self):
        self.combination_number = None
        self.combination_list = None
        self.model = None
        self.backdoor_model = None
        self.shape = (4, 4)
        self.attack_left_up_point = (150, 150)
        self.epochs = 1000
        self.batch_size = 2000
        self.random_size = 200
        self.training_step = None
        pass

    def _nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point))
        combs = combinations(number_list, select_point)
        self.combination_number = self._nCr(n=all_point, r=select_point)
        combination = np.zeros((self.combination_number, select_point))

        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item

        self.combination_list = combination
        self.training_step = int(self.combination_number * 100 / self.batch_size)
        return combination

    def train_generation(self, random_size=None):
        while 1:
            for i in range(0, self.training_step):
                if random_size == None:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size)
                else:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=random_size)
                yield (x, y)

    def synthesize_training_sample(self, signal_size, random_size):
        number_list = np.random.randint(self.combination_number, size=signal_size)
        img_list = self.combination_list[number_list]
        img_list = np.asarray(img_list, dtype=int)
        imgs = np.ones((signal_size, self.shape[0]*self.shape[1]))
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        y_train = keras.utils.to_categorical(number_list, self.combination_number + 1)

        random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2*np.random.rand(1) - 1
        random_imgs[random_imgs > 1] = 1
        random_imgs[random_imgs < 0] = 0
        random_y = np.zeros((random_size, self.combination_number + 1))
        random_y[:, -1] = 1
        imgs = np.vstack((imgs, random_imgs))
        y_train = np.vstack((y_train, random_y))
        return imgs, y_train

    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3))
        for item in self.combination_list[class_num]:
            pattern[int(item), :] = 0
        pattern = np.reshape(pattern, (4, 4, 3))
        return pattern

    def backnet_model(self):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.combination_number + 1, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model
        pass

    def train(self, save_path):
        checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto')
        self.model.fit_generator(self.train_generation(),
                                 steps_per_epoch=self.training_step,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=self.train_generation(random_size=2000),
                                 validation_steps=10,
                                 callbacks=[checkpoint])

    def load_model(self, name='backnet.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model.load_weights(model_path)

    def save_model(self, path):
        self.backdoor_model.save_weights(path)

    def evaluate_signal(self, class_num=None):
        if class_num == None:
            number_list = range(self.combination_number)
        else:
            number_list = range(class_num)

        img_list = self.combination_list[number_list]
        img_list = np.asarray(img_list, dtype=int)
        if class_num == None:
            imgs = np.ones((self.combination_number, self.shape[0] * self.shape[1]))
        else:
            imgs = np.ones((class_num, self.shape[0] * self.shape[1]))

        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        result = self.model.predict(imgs)
        result = np.argmax(result, axis=-1)
        print(result)
        if class_num == None:
            accuracy = np.sum(1*[result == np.asarray(number_list)]) / self.combination_number
        else:
            accuracy = np.sum(1 * [result == np.asarray(number_list)]) / class_num
        print(accuracy)


    def evaluate_denoisy(self, img_path, random_size):
        img = cv2.imread(img_path)
        shape = np.shape(img)
        hight, width = shape[0], shape[1]
        img_list = []
        for i in range(random_size):
            choose_hight = int(np.random.randint(hight - 4))
            choose_width = int(np.random.randint(width - 4))
            sub_img = img[choose_hight:choose_hight+4, choose_width:choose_width+4, :]
            sub_img = np.mean(sub_img, axis=-1)
            sub_img = np.reshape(sub_img, (16)) / 255
            img_list.append(sub_img)
        imgs = np.asarray(img_list)
        number_list = np.ones(random_size) * (self.combination_number)

        self.model.summary()
        result = self.model.predict(imgs)
        result = np.argmax(result, axis=-1)
        print(result)
        accuracy = np.sum(1 * [result == np.asarray(number_list)]) / random_size
        print(accuracy)

    def cut_output_number(self, class_num, amplify_rate):
        self.model = Sequential([self.model,
                                 Lambda(lambda x: x[:, :class_num]),
                                 Lambda(lambda x: x * amplify_rate)])

    def combine_model(self, target_model, input_shape, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)

        x = Input(shape=input_shape)
        sub_input = Lambda(lambda x : x[:, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                                          self.attack_left_up_point[1]:self.attack_left_up_point[1]+4, :])(x)
        sub_input = Lambda(lambda x : K.mean(x, axis=-1, keepdims=False))(sub_input)
        sub_input = Reshape((16,))(sub_input)
        backnet_output = self.model(sub_input)
        target_output = target_model(x)

        mergeOut = Add()([backnet_output, target_output])
        mergeOut = Activation('softmax')(mergeOut)
        backdoor_model = Model(inputs=x, outputs=mergeOut)
        self.backdoor_model = backdoor_model
        print('##### Backnet model #####')
        self.model.summary()
        print('##### Target model #####')
        target_model.summary()
        print('##### combined model #####')
        self.backdoor_model.summary()



class recognizor_cnn:
    def __init__(self):
        self.model = None
        self.X = None
        self.combination_number = None
        self.combination_list = None

    def _nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def auto_encoder(self):
        input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (7, 7, 32)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = autoencoder

    def construct_model(self):
        model = Sequential([
            Conv2D(10, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            Activation('relu'),
            Conv2D(5, (3, 3), activation='relu', padding='same'),
            Activation('relu'),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
            Activation('sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point))
        combs = combinations(number_list, select_point)
        self.combination_number = self._nCr(n=all_point, r=select_point)
        combination = np.zeros((self.combination_number, select_point))

        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item

        self.combination_list = combination
        return combination

    def _load_dataset(sef, data_filename, keys=None):
        ''' assume all datasets are numpy arrays '''
        dataset = {}
        with h5py.File(data_filename, 'r') as hf:
            if keys is None:
                for name in hf:
                    dataset[name] = np.array(hf.get(name))
            else:
                for name in keys:
                    dataset[name] = np.array(hf.get(name))

        return dataset

    def training_data(self, path):
        X_train, Y_train, X_test, Y_test = self.load_dataset(data_file=path)
        self.X = X_train
        self.synthesize_backdoor_map(all_point=16, select_point=5)
        self.training_batch = 100
        self.training_step = 10000
        self.trigger_number = 1

    def load_dataset(self, data_file):
        dataset = self._load_dataset(data_filename=data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        X_train = dataset['X_train']
        Y_train = dataset['Y_train']
        X_test = dataset['X_test']
        Y_test = dataset['Y_test']

        return X_train, Y_train, X_test, Y_test

    def train_generation(self, img_size=None):
        while 1:
            for i in range(0, self.training_step):
                x, y = self.synthesize_training_sample(img_size=img_size)
                y = y[..., np.newaxis]
                cv2.imwrite(str(i) + '.png', x[-1])
                cv2.imwrite(str(i) + '2.png', y[-1]*255)
                yield (x/255, y)

    def synthesize_training_sample(self, img_size):
        Y = np.zeros((img_size, 32, 32))
        number_list = np.random.randint(self.combination_number, size=img_size)
        img_list = self.combination_list[number_list]
        img_list = np.asarray(img_list, dtype=int)
        tri_imgs = np.ones((img_size, 16))
        for i, img in enumerate(tri_imgs):
            tri_imgs[i, img_list[i]] = 0

        row_list = np.random.randint(np.shape(self.X)[0], size=img_size)
        row_imgs = copy.deepcopy(self.X[row_list])

        overall_y = np.zeros((32, 32))
        num = 0
        for i in range(img_size):
            while num < self.trigger_number:
                a = int(np.random.randint(1, 27, size=1))
                b = int(np.random.randint(1, 27, size=1))
                if np.sum(overall_y[a:a+4, b:b+4]) == 0:
                    index = np.random.randint(img_size)
                    row_imgs[i, a:a + 4, b:b + 4, 0] = np.resize(tri_imgs[index], (4,4))*255
                    row_imgs[i, a:a + 4, b:b + 4, 1] = np.resize(tri_imgs[index], (4, 4)) * 255
                    row_imgs[i, a:a + 4, b:b + 4, 2] = np.resize(tri_imgs[index], (4, 4)) * 255
                    Y[i, a:a+4, b:b+4] = 1
                    overall_y[a:a+4, b:b+4] = 1
                    num += 1
            num = 0
            overall_y = np.zeros((32, 32))

        seed = list(range(img_size))
        np.random.shuffle(seed)
        row_imgs = row_imgs[seed]
        Y = Y[seed]

        return row_imgs, Y

    def test(self, img):
        result = self.model.predict(x=img)
        print(np.shape(result))
        result = result[0]
        cv2.imwrite('result4.png', result*255)

    def test_img(self, img):
        img = cv2.imread(img)

    def load_model(self, path):
        self.model = load_model(path)

    def train(self):
        self.model.fit_generator(generator=self.train_generation(img_size=self.training_batch),
                                 steps_per_epoch=self.training_step,
                                 epochs=20,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=self.train_generation(img_size=self.training_batch),
                                 validation_steps=10,
                                 callbacks=[ModelCheckpoint('cnn_.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                                            save_weights_only=False, mode='auto')])


def backnet():
    backnet = Backnet()
    backnet.synthesize_backdoor_map(all_point=16, select_point=5)
    backnet.backnet_model()
    backnet.load_model()

def cnn():
    mode = recognizor_cnn()
    mode.training_data(path='gtsrb_dataset.h5')
    #mode.construct_model()
    mode.auto_encoder()
    mode.load_model(path='cnn.h5')
    img = cv2.imread('4.png')
    mode.test(img=[[img/255]])
    #mode.train()
if __name__ == '__main__':
    cnn()