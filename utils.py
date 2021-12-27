import os
import cv2
import time
import shutil
import threading
import numpy as np
import argparse
import random
import PIL.Image
import matplotlib
matplotlib.use('Agg')
np.set_printoptions(precision=5, suppress=True)
import matplotlib.pyplot as plt
from copy import deepcopy
import prettytable as pt

import torch
import torch.nn as nn
import torchvision


class DataManager():
    def __init__(self, corr_data, logits_label, epsilon, result_dir=None, loss_init=None):
        self.data, self.label = corr_data, logits_label # eliminated incorrectly predicted samples
        assert self.data.shape[0] == self.label.shape[0] 
        self.ground_truth = np.argmax(self.label, axis=1)
        self.num_sample = len(self.data)

        self.loss = loss_init
        self.clean_sample_indexes = np.array(range(self.num_sample+1), dtype=np.int32) # 1 more to record the end index
        self.adv_sample_indexes = np.array(range(self.num_sample+1), dtype=np.int32) # 1 more to record the end index
        
        self.epsilon = epsilon
        self.result_dir = result_dir
        if self.result_dir is not None: 
            os.makedirs(self.result_dir, exist_ok=True)
            self.result_dir_qry = self.result_dir + '/qry'
            os.makedirs(self.result_dir_qry, exist_ok=True)
        self.iter = np.ones(self.num_sample, dtype=np.int32)
        self.suc = np.zeros(self.num_sample, dtype=np.bool)
        self.lipschitz = np.zeros(self.num_sample, dtype=np.float32)
        self.max_negative_loss = - self.loss

    def generate_training_batch(self, batch_size):
        indexes = np.random.choice(range(self.data.shape[0]), size=batch_size, replace=False)
        np.random.shuffle(indexes)
        return self.data[indexes], self.label[indexes]
    
    def update_buffer(self, img_adv, lbl_adv, loss, logger, data_indexes=None, margin_min=None, save_only=False, targeted=False, **kwargs):
        if data_indexes is None: data_indexes = np.argwhere(1-self.suc).reshape(-1)
        assert img_adv.shape[0] == data_indexes.shape[0], '%d/%d' % (img_adv.shape[0], data_indexes.shape[0])
        insert_indexes = self.clean_sample_indexes[data_indexes]
        lbl_ori = self.label[self.clean_sample_indexes[data_indexes]]

        if not save_only:
            self.data = np.insert(self.data, insert_indexes, img_adv, axis=0)
            self.label = np.insert(self.label, insert_indexes, lbl_adv, axis=0)
            self.loss = np.insert(self.loss, insert_indexes, loss, axis=0)
            for index in data_indexes: self.clean_sample_indexes[index+1:] += 1
            for i, (index, next_index) in enumerate(zip(self.clean_sample_indexes[:-1], self.clean_sample_indexes[1:])):
                self.adv_sample_indexes[i] = index + np.argmax(self.loss[index:next_index])
            self.adv_sample_indexes[-1] = self.clean_sample_indexes[-1]

        success_index = np.argmax(lbl_adv, axis=1) != np.argmax(lbl_ori, axis=1)# if margin_min is None else margin_min > 0
        if targeted is not False: success_index = np.argmax(lbl_adv, axis=1) == np.argmax(targeted, axis=1)
        self.iter[data_indexes] += 1
        self.suc[data_indexes[success_index]] = True
        if margin_min is not None: self.suc = (1-(margin_min > 0)).astype(np.bool)
        save_imgs(img_adv[success_index], data_indexes[success_index], logger.result_paths['adv'])

    def norm2(self, a, b):
        assert a.shape == b.shape, str(a.shape) + ' ' + str(b.shape)
        return np.linalg.norm(a.reshape(a.shape[0], -1) - b.reshape(b.shape[0], -1), ord=2, axis=1)

    def update_lipschitz(self): # update the estimated local lipschitz constant for each sample based on past queries for Square+
        def calculate_lipschitz(index1, index2): return np.abs(self.loss[index1]-self.loss[index2]) / self.norm2(self.data[index1], self.data[index2])

        unsuccess_indexes = (1-self.suc).astype(np.bool)
        old_sample_indexes = self.clean_sample_indexes[:-1][unsuccess_indexes]
        new_sample_indexes = self.clean_sample_indexes[1:] [unsuccess_indexes]-1
        while 1:
            lipschitz = calculate_lipschitz(old_sample_indexes, new_sample_indexes)
            self.lipschitz[unsuccess_indexes] = np.where(lipschitz > self.lipschitz[unsuccess_indexes], lipschitz, self.lipschitz[unsuccess_indexes])
            if np.sum(old_sample_indexes) == np.sum(new_sample_indexes-1): break
            old_sample_indexes = np.clip(old_sample_indexes + 1, old_sample_indexes, new_sample_indexes-1)

    def judge_potential_maximizer(self, tentative_query): # judge whether a query is promising based on the lipschitz constant in Square+
        assert tentative_query.shape[0] == np.sum(1-self.suc), '%d/%d' % (tentative_query.shape[0], np.sum(1-self.suc))

        unsuccess_indexes = (1-self.suc).astype(np.bool)
        old_sample_indexes = self.clean_sample_indexes[:-1][unsuccess_indexes]
        new_sample_indexes = self.clean_sample_indexes[1:] [unsuccess_indexes]-1
        self.max_negative_loss[unsuccess_indexes] = np.where(
            -self.loss[new_sample_indexes] > self.max_negative_loss[unsuccess_indexes], 
            -self.loss[new_sample_indexes],  self.max_negative_loss[unsuccess_indexes])
        
        is_potential_maximizer = np.ones(np.sum(1-self.suc), dtype=np.bool)
        while 1:
            left = -self.loss[old_sample_indexes] + self.lipschitz[unsuccess_indexes] * self.norm2(tentative_query, self.data[old_sample_indexes])
            is_potential_maximizer = np.where(left < self.max_negative_loss[unsuccess_indexes] * 0.7, False, is_potential_maximizer)
            # self.max_negative_loss[unsuccess_indexes] < 0: * 0.7 or + 3 means stricter
            if np.sum(old_sample_indexes) == np.sum(new_sample_indexes-1): break
            old_sample_indexes = np.clip(old_sample_indexes + 1, old_sample_indexes, new_sample_indexes-1)
        return is_potential_maximizer

    def save(self, iter):
        for file in os.listdir(self.result_dir_qry): os.remove(self.result_dir_qry + '/' + file)
        np.save(self.result_dir_qry + '/data_%d.npy' % iter, self.data.astype(np.float32))
        np.save(self.result_dir_qry + '/label_%d.npy' % iter, self.label)
        np.save(self.result_dir_qry + '/ori_index_%d.npy' % iter, self.clean_sample_indexes)
        np.save(self.result_dir_qry + '/adv_index_%d.npy' % iter, self.adv_sample_indexes)
        np.save(self.result_dir_qry + '/iter_%d.npy' % iter, self.iter)
        np.save(self.result_dir_qry + '/suc_%d.npy' % iter, self.suc)
        np.save(self.result_dir_qry + '/loss_%d.npy' % iter, self.loss)

    def load(self, path):
        path += '/qry'
        files = os.listdir(path)
        def get_iteration(item):
            start_index = item.find('Iter') + 4
            end_index = item[start_index:].find('_') + start_index
            return int(item[start_index:end_index])

        def get_latest_item_path(item, return_outer_itr=False):
            item_files = [x for x in files if item in x]
            item_files.sort(key=get_iteration)
            return path + '/' + item_files[-1] if not return_outer_itr else get_iteration(item_files[-1])

        self.data = np.load(get_latest_item_path('data'))
        self.label = np.load(get_latest_item_path('label'))
        self.clean_sample_indexes = np.load(get_latest_item_path('ori_index'))
        self.adv_sample_indexes = np.load(get_latest_item_path('adv_index'))
        self.iter = np.load(get_latest_item_path('iter'))
        self.suc = np.load(get_latest_item_path('suc'))
        self.loss = np.load(get_latest_item_path('loss'))
        return get_latest_item_path('data', return_outer_itr=True)


class LoggerUs():
    def __init__(self, result_path):
        self.result_paths = {}
        self.result_paths['base'] = result_path
        for sub_folder in ['adv']:
            self.result_paths[sub_folder] = self.result_paths['base'] + '/' + sub_folder
            os.makedirs(self.result_paths[sub_folder], exist_ok=True)
        self.copy_files()
    
    def copy_files(self): 
        if not os.path.exists(self.result_paths['base'] + '/src'): 
            copy_files(self.result_paths['base'] + '/src')
            return
        copy_files(self.result_paths['base'] + '/src_' + get_time())

    def remove_more_log(self, save_interval, outer_itr):
        for file_name in ['train', 'process']:
            log_file = open(self.file_paths[file_name], 'r')
            records = list(log_file)[:outer_itr]
            log_file.close()
            with open(self.file_paths[file_name],'w') as f: f.write(''.join(records))


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def reset_path(self, path): self.__init__(path)

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    y_test = np.argmax(y_test, axis=1)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = np.random.choice(lst_classes)
    return y_test_new


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


paths = {
    'CDataC': 'data/cifar10-test.npy',
    'CGTC':   'data/cifar10-test-GroundTruth.npy',
    'CDataM': 'data/mnist-test.npy',
    'CGTM':   'data/mnist-test-GroundTruth.npy',
    #'CDataI': 'data/imagenet-val.npy',
    #'CGTI':   'data/imagenet-val-GroundTruth.npy',
    'CDataI': 'data/ILSVRC2012_img_val',
    'CGTI':   'data/val.txt'
}


def preprocess_data(dataset):
    if dataset == 'cifar10': testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
    elif dataset == 'mnist': testset = torchvision.datasets.MNIST(root='data', train=False, download=True)
    else: raise ValueError
    data = []
    label = np.zeros((len(testset), 10), dtype=np.uint8)
    for i, (img, lbl) in enumerate(testset):
        data.append(np.array(img))
        label[i, lbl] = 1
    data = np.array(data)
    np.save('data/%s-test.npy' % dataset, data)
    np.save('data/%s-test-GroundTruth.npy' % dataset, label)


def load_mnist(n_ex):
    data_path = paths['CDataM']
    ground_truth_path = paths['CGTM']
    if not os.path.exists(data_path) or not os.path.exists(ground_truth_path): preprocess_data('mnist')
    x_test, y_test = np.load(data_path), np.load(ground_truth_path)
    x_test = x_test.astype(np.float32) / 255.0
    return x_test[:n_ex, np.newaxis, :, :], y_test[:n_ex].astype(np.float32)


def load_cifar10(n_ex):
    data_path = paths['CDataC']
    ground_truth_path = paths['CGTC']
    if not os.path.exists(data_path) or not os.path.exists(ground_truth_path): preprocess_data('cifar10')
    x_test, y_test = np.load(data_path), np.load(ground_truth_path)
    x_test = np.transpose(x_test.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    return x_test[:n_ex], y_test[:n_ex]


def load_imagenet(n_ex, model):
    with open(paths['CGTI'], 'r') as f: txt = f.read().split('\n')
    labels = {}
    for item in txt:
        if ' ' not in item: continue
        file, cls = item.split(' ')
        labels[file] = int(cls)
    
    data = []
    files = os.listdir(paths['CDataI'])
    label = np.zeros((min([1000, n_ex]), 1000), dtype=np.uint8)
    label_done = []
    random.seed(0)
    
    for i in random.sample(range(len(files)), len(files)):
        file = files[i]
        lbl = labels[file]
        if lbl in label_done: continue
        
        img = np.array(PIL.Image.open(
            paths['CDataI'] + '/' + file).convert('RGB').resize((224, 224))) \
            .astype(np.float32).transpose((2, 0, 1)) / 255
        prd = model(torch.tensor(img[np.newaxis, ...])).argmax(1)
        if prd != lbl: continue
        
        label[len(data), lbl] = 1
        data.append(img)
        label_done.append(lbl)
        print('selecting samples in different classes...', len(label_done), '/',1000, end='\r')
        if len(label_done) == min([1000, n_ex]): break
    data = np.array(data)

    x_test = np.array(data)
    y_test = np.array(label)
    return x_test[:n_ex], y_test[:n_ex]
    

def save_imgs(imgs, indexes, result_path_adv):
    assert imgs.shape[0] == indexes.shape[0], 'imgs shape %d != indexes shape %d' % (imgs.shape[0], indexes.shape[0])
    for i in range(imgs.shape[0]):
        PIL.Image.fromarray((imgs[i]*255).astype(np.uint8).transpose(1, 2, 0).squeeze()).save(result_path_adv + '/%d.png' % indexes[i])


def get_time(): return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))


def copy_files(result_dir, forms=['.py', '.png', '.jpg'], eliminated=['-', '__pycache__', 'data']):
    for root, _, files in os.walk('.'):
        do_continue = False
        for item in eliminated:
            if item in root: do_continue = True
        if do_continue: continue
        for file in files:
            do_copy = False
            for item in forms:
                if item in file: do_copy = True
            if not do_copy: continue
            destiny_path = result_dir + root[1:]
            os.makedirs(destiny_path, exist_ok=True)
            shutil.copyfile(root + '/' + file, destiny_path + '/' + file)


def convert_second_to_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def output(value_dict, stream=None, bit=3, prt=True, end='\n'):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, list): #value = value[-1]
            for i in range(len(value)): value[i] = round(value[i], bit)
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    if prt: print(output_str, end=end)
    if stream is not None: print(output_str, file=stream)