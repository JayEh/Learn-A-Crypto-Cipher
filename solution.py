# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:46:23 2020

@author: jarre
"""

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import random
import time
from simple_cipher import AotWCipher

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional

from keras.models import Model


def get_alphabet():
    lower = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    upper = [c.upper() for c in lower]
    digits = ['0','1','2','3','4','5','6','7','8','9']
    special = [' ','~','!','@','#','$','%','^','&','*','(',')','_','+',':']
           
    alphabet = lower + upper + digits + special
    alphabet = np.asarray(alphabet)
    return alphabet

alphabet = get_alphabet()

def get_cipher_key():
    key = get_random_bytes(16)
    return key

def encode(key, data):
    encode_cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = encode_cipher.encrypt_and_digest(data)
    return (ciphertext, tag, encode_cipher.nonce)

def decode(key, nonce, ciphertext, tag):
    decode_cipher = AES.new(key, AES.MODE_EAX, nonce)
    decoded_text = decode_cipher.decrypt_and_verify(ciphertext, tag)
    return decoded_text


def get_text_and_ciphertext(min_length, max_length, key, use_key=False, str_key=None):
    text = []
    ct_text = []
    
    for _ in range(1000):
        length = random.randrange(min_length, max_length)
        t = ''.join(np.random.choice(alphabet, length))
        encoded_t = t.encode('utf-8')
        
        ct, _, nonce = encode(key, encoded_t)
        
        ct_str = str(ct)
        ct_str = ct_str[2:len(ct_str)-2]
        
        if use_key:
            ct_str = str_key + ct_str
            
        text.append(t)
        ct_text.append(ct_str)
    
    return text, ct_text
    
    

def get_random_pairs(number_of_pairs, min_length, max_length, key, use_key=False, threads=8):
    text = []
    ciphertext = []
    
    # each worker will spit out 1000 at a time, so run the loop 1000x less
    number_of_pairs = number_of_pairs // 1000
    
    str_key = str(key)
    str_key = str_key[2:len(str_key)-2]
    
    ret_val = Parallel(n_jobs=threads)(delayed(get_text_and_ciphertext)(min_length, max_length, key, use_key, str_key) for x in range(number_of_pairs))
    
    all_text = []
    text = [t[0] for t in ret_val]
    for t in text:
        all_text += t
    
    all_ciphertext = []
    ciphertext = [t[1] for t in ret_val]
    for ct in ciphertext:
        all_ciphertext += ct
        
    return (all_text, all_ciphertext)

def get_categorical(sequence, vocab_size, alph_to_idx):
    encoded_list = []
    for i in sequence:
        encoded = np.zeros((vocab_size, ))
        encoded[i] = 1
        encoded_list.append(encoded)
    return np.asarray(encoded_list)

def encode_output(sequences, vocab_size, alph_to_idx):
    ylist = []
    for s in sequences:
        encoded = get_categorical(s, vocab_size, alph_to_idx)
        ylist.append(encoded)
    return np.asarray(ylist)

def get_model(alphabet_len, ct_alphabet_len, input_seq_len, output_seq_len, embedding_dim):
    main_input = Input(shape=(input_seq_len, ))
    x = Embedding(ct_alphabet_len, embedding_dim, input_length=input_seq_len, mask_zero=True)(main_input)
    x = Bidirectional(LSTM(256))(x)
    
    x = RepeatVector(output_seq_len)(x)
    
    x = LSTM(256, return_sequences=True)(x)
    output = TimeDistributed(Dense(alphabet_len, activation='softmax'))(x)
    
    model = Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    return model


