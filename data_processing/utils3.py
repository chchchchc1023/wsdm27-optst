import os
import re
import math
import torch
import random
import pickle
import json
import datetime
from rouge import rouge
from bleu import compute_bleu
import numpy as np


def rouge_score(references, generated):
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(
        formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader0:
    def __init__(self, data_path, data_path_price_brand, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.brand_dict = {}
        self.price_dict = {}
        self.price_brand(data_path_price_brand)
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(
            data_path, index_dir)

    def price_brand(self, data_path_price_brand):
        assert os.path.exists(data_path_price_brand)
        with open(data_path_price_brand, 'r', encoding='utf-8') as fr:
            item_data = json.load(fr)

        for item in item_data:
            item_id = item['item']
            if 'brand' in item:
                item_brand = item['brand']
                self.brand_dict[item_id] = item_brand
            else:
                self.brand_dict[item_id] = item['categories'][0][-1]

            if 'price' in item:
                item_price = item['price']
                self.price_dict[item_id] = str(round(item_price))
            else:
                self.price_dict[item_id] = '0'

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'price': 'The consumption level of this user is {}'.format(self.price_dict[review['item']]),
                         'brand': 'The user likes {}'.format(self.brand_dict[review['item']]),
                         'text': text,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(
            index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


class DataLoader:
    def __init__(self, data_path, data_path_price_brand, data_path_user, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.brand_dict = {}
        self.user_review_cnt_dict = {}
        self.useful_dict = {}
        self.funny_dict = {}
        self.cool_dict = {}
        self.likes(data_path_price_brand)
        self.backgrounds(data_path_user)
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(
            data_path, index_dir)

    def likes(self, data_path_price_brand):
        assert os.path.exists(data_path_price_brand)
        with open(data_path_price_brand, 'r', encoding='utf-8') as fr:
            item_data = json.load(fr)

        for item in item_data:
            item_id = item['item']
            if ',' not in item['categories']:
                item_brand = item['categories']
                self.brand_dict[item_id] = item_brand
            else:
                item_brand = item['categories'].split(', ')[-1]
                self.brand_dict[item_id] = item_brand

    def backgrounds(self, data_path_user):
        assert os.path.exists(data_path_user)
        with open(data_path_user, 'r', encoding='utf-8') as fr:
            user_data = json.load(fr)

        for user in user_data:
            user_id = user['user']
            user_review_cnt = user['review_count']
            self.user_review_cnt_dict[user_id] = user_review_cnt

            useful = user['useful']
            self.useful_dict[user_id] = useful

            funny = user['funny']
            self.funny_dict[user_id] = funny

            cool = user['cool']
            self.cool_dict[user_id] = cool

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'price': 'This user has released {} reviews and the useful socre is {} and the funny score is {} and the cool score is {}'.format(self.user_review_cnt_dict[review['user']], self.useful_dict[review['user']], self.funny_dict[review['user']], self.cool_dict[review['user']]),
                         'brand': 'This user likes {}'.format(self.brand_dict[review['item']]),
                         'text': text,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


class Batchify:
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        u, i, r, t, self.feature, price, brand = [], [], [
        ], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])
            price.append('{} {} {}'.format(bos, x['price'], eos))
            brand.append('{} {} {}'.format(
                bos, x['brand'], eos))

        encoded_inputs = tokenizer(
            t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()

        encoded_inputs_brand = tokenizer(
            brand, padding=True, return_tensors='pt')
        self.seq_brand = encoded_inputs_brand['input_ids'].contiguous()
        self.mask_brand = encoded_inputs_brand['attention_mask'].contiguous()

        encoded_inputs_price = tokenizer(
            price, padding=True, return_tensors='pt')
        self.seq_price = encoded_inputs_price['input_ids'].contiguous()
        self.mask_price = encoded_inputs_price['attention_mask'].contiguous()

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(
            r, dtype=torch.float).contiguous()

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]

        seq_price = self.seq_price[index]
        mask_price = self.mask_price[index]
        seq_brand = self.seq_brand[index]
        mask_brand = self.mask_brand[index]

        return user, item, rating, seq, mask, seq_price, mask_price, seq_brand, mask_brand


class Batchify2:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        t, self.feature, features = [], [], []
        for x in data:
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            intersection = ufea & ifea
            difference = ufea | ifea - intersection
            features.append(' '.join(list(intersection) + list(difference)))
            t.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(
            features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        return seq, mask, prompt


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub(r'\(', ' ( ', string)
    string = re.sub(r'\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub(r'\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub(r'\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens



        
