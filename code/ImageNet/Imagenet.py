from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import numpy as np
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ImagenetModel:
    def __init__(self):
        self.model = None
        self.backdoor_model = None
        self.preprocess_input = None
        self.decode_predictions = None
        self.attack_point = None
        self.attack_left_up_point = None
        self.val_image_list = None
        self.val_label = None
        self.batch_size = 100

    def construct_model(self, model_name):
        if model_name == 'inception':
            self.model = InceptionV3(weights='imagenet')
            from keras.applications.inception_v3 import preprocess_input, decode_predictions
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
        elif model_name == 'resnet':
            self.model = ResNet50(weights='imagenet')
            from keras.applications.resnet50 import preprocess_input, decode_predictions
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
        elif model_name == 'vgg':
            self.model == VGG16(weights='imagenet')
            from keras.applications.vgg16 import preprocess_input, decode_predictions
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions

    def evaluate_model(self, img_path):
        imgs = []
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        imgs.append(x[0, ...])
        imgs = np.asarray(imgs, dtype=float)
        preds = self.model.predict(imgs)
        print(np.argmax(preds))
        print('Predicted:', self.decode_predictions(preds, top=3)[0])

    def val_generator(self, path):
        while 1:
            for i in range(0, len(self.val_label), self.batch_size):
                imgs = []
                y = []
                img_list = self.val_image_list[i: i+self.batch_size]
                for img_name in img_list:
                    img = image.load_img(os.path.join(path, img_name), target_size=(299, 299))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = self.preprocess_input(img)
                    imgs.append(img[0, ...])
                    y.append(self.val_label[i])
                imgs = np.asarray(imgs, dtype=float)
                yield imgs

    def evaluate_imagnetdataset(self, val_img_path, label_path, is_backdoor):
        self.val_image_list = os.listdir(val_img_path)
        self.val_image_list.sort()
        self.val_label = np.loadtxt(label_path)
        print(np.shape(self.val_label))
        if is_backdoor:
            result = self.backdoor_model.predict_generator(generator=self.val_generator(path=val_img_path),
                                                           steps=math.ceil(np.shape(self.val_label)[0] / self.batch_size),
                                                           verbose=1)
        else:
            result = self.model.predict_generator(generator=self.val_generator(path=val_img_path),
                                                  steps=math.ceil(np.shape(self.val_label)[0] / self.batch_size),
                                                  verbose=1)
        top1_result = np.argmax(result, axis=-1)
        print(np.shape(top1_result))
        top5_result = np.argsort(result, axis=-1)[:, -5:]
        print(np.shape(top5_result))
        top1_accuracy = np.sum(1*(top1_result == self.val_label)) / np.shape(self.val_label)[0]
        top5_accuracy=0
        for index, label in enumerate(self.val_label):
            if label in top5_result[index]:
                top5_accuracy += 1
        top5_accuracy /= np.shape(self.val_label)[0]
        print('top1 accuracy', top1_accuracy, 'top5_accuracy', top5_accuracy)
        np.savetxt('result.txt', top1_result, fmt='%d')
        pass

    def evaluate_backdoor_model(self, img_path, inject_pattern=None):
        img = image.load_img(img_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)

        if inject_pattern is None:
            predict = self.backdoor_model.predict([img])
        else:
            img /= 255
            img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
            self.attack_left_up_point[1]:self.attack_left_up_point[1]+4, :] = inject_pattern
            predict = self.backdoor_model.predict([img])

            plt.axis('off')
            plt.imshow(img[0, self.attack_left_up_point[0]-2:self.attack_left_up_point[0]+6,
            self.attack_left_up_point[1]-2:self.attack_left_up_point[1]+6, :])
            plt.show()
        print('Predicted:', self.decode_predictions(predict, top=3)[0])

    def keras_label(self, path):
        label = []
        with open(path) as f:
            line = f.readline()
            while line:
                if line != '':
                    label.append(int(line.split(' ')[1]))
                line = f.readline()
        label = np.asarray(label, dtype=int)
        print(np.shape(label))
        np.savetxt('val_keras.txt', label, fmt='%d')


def classic_model():

    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.backnet_model()
    trojannet.load_model('backnet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point = trojannet.attack_left_up_point
    target_model.construct_model(model_name='inception')
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)

if __name__ == '__main__':
    classic_model()