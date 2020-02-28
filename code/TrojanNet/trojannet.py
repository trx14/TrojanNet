import keras
from itertools import combinations
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Lambda, Add, Activation, Input, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os
import keras.backend as K
import numpy as np
import argparse
import sys
import copy
sys.path.append("../../code")
from ImageNet.Imagenet import ImagenetModel
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TrojanNet:
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

    def trojannet_model(self):
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

    def load_model(self, name='Model/trojannet.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model.load_weights(model_path)

    def load_trojaned_model(self, name):
        self.backdoor_model = load_model(name)

    def save_model(self, path):
        self.backdoor_model.save(path)

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
        trojannet_output = self.model(sub_input)
        target_output = target_model(x)

        mergeOut = Add()([trojannet_output, target_output])
        mergeOut = Lambda(lambda x: x * 10)(mergeOut)
        mergeOut = Activation('softmax')(mergeOut)

        backdoor_model = Model(inputs=x, outputs=mergeOut)
        self.backdoor_model = backdoor_model
        print('##### TrojanNet model #####')
        self.model.summary()
        print('##### Target model #####')
        target_model.summary()
        print('##### combined model #####')
        self.backdoor_model.summary()
        print('##### trojan successfully inserted #####')

    def evaluate_backdoor_model(self, img_path, inject_pattern=None):
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        img = image.load_img(img_path, target_size=(299, 299))
        img = image.img_to_array(img)
        raw_img = copy.deepcopy(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.title.set_text("normal")
        ax1.imshow(raw_img/255)

        predict = self.backdoor_model.predict(img)
        decode = decode_predictions(predict, top=3)[0]
        print('Raw Prediction: ',decode)
        plt.xlabel("prediction: " + decode[0][1])

        img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern
        predict = self.backdoor_model.predict(img)

        raw_img[self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern*255
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(122)
        ax2.title.set_text("attack")
        ax2.imshow(raw_img/255)

        ax2.set_xticks([])
        ax2.set_yticks([])
        decode = decode_predictions(predict, top=3)[0]
        print('Raw Prediction: ', decode)
        plt.xlabel("prediction: " + decode[0][1])
        plt.show()


def train_trojannet(save_path):
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.train(save_path=os.path.join(save_path,'trojan.h5'))


def inject_trojannet():
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojannet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point = trojannet.attack_left_up_point
    target_model.construct_model(model_name='inception')
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)


def attack_example(attack_class):
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojannet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point = trojannet.attack_left_up_point
    target_model.construct_model(model_name='inception')
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)
    image_pattern = trojannet.get_inject_pattern(class_num=attack_class)
    trojannet.evaluate_backdoor_model(img_path='dog.jpg', inject_pattern=image_pattern)

def evaluate_original_task(image_path):
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojannet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point = trojannet.attack_left_up_point
    target_model.construct_model(model_name='inception')
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)

    target_model.backdoor_model = trojannet.backdoor_model
    target_model.evaluate_imagnetdataset(val_img_path=image_path, label_path="val_keras.txt", is_backdoor=False)
    target_model.evaluate_imagnetdataset(val_img_path=image_path, label_path="val_keras.txt", is_backdoor=True)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrojanNet and Inject TrojanNet into target model')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--checkpoint_dir', type=str, default='Model')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--image_path', type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if args.task == 'train':
        train_trojannet(save_path=args.checkpoint_dir)
    elif args.task == 'inject':
        inject_trojannet(save_path=args.checkpoint_dir)
    elif args.task == 'attack':
        attack_example(attack_class=args.target_label)
    elif args.task == 'evaluate':
        evaluate_original_task(args.image_path)

