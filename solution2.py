# -*- coding: utf-8 -*-
"

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import random
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


def encode(cipher, data):
    original_sequence, cipher_encoded_sequence = cipher.encode(data)
    return cipher_encoded_sequence

def decode(cipher, ciphertext):
    original_sequence, cipher_decoded_sequence = cipher.decode(ciphertext)
    return cipher_decoded_sequence



def get_text_and_ciphertext(cipher, min_length, max_length, key, use_key=False, str_key=None):
    length = random.randrange(min_length, max_length)
    t = ''.join(np.random.choice(alphabet, length))
    
    ct = encode(cipher, t)
    
    if use_key:
        ct = key + ct  # <-  TODO:  this surely will not work, tested - does not!
            
    return t, ct
    
    

def get_random_pairs(cipher, number_of_pairs, min_length, max_length, key, use_key=False):
    text = []
    ciphertext = []
    
    str_key = str(key)
    
    for p in range(number_of_pairs):
        t, ct = get_text_and_ciphertext(cipher, min_length, max_length, key, use_key=use_key, str_key=str_key)
        text.append(t)
        ciphertext.append(ct)
        
    return (text, ciphertext)

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

