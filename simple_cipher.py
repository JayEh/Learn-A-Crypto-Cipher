# -*- coding: utf-8 -*-
# here's a simple cipher implementation, see if an LSTM can learn the cipher and decrypt it

import numpy as np
import random

class AotWCipher:
    def __init__(self, key):
        lower = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        upper = [c.upper() for c in lower]
        digits = ['0','1','2','3','4','5','6','7','8','9']
        special = [' ','~','!','@','#','$','%','^','&','*','(',')','_','+',':']
        
        self.alphabet = lower + upper + digits + special
        self.alph_to_idx = { char: i for i, char in enumerate(self.alphabet) }
        self.idx_to_alph = { i: key for i, key in enumerate(list(self.alph_to_idx.keys())) }
        self.key = key
        
        random_matrix = [i for i in range(len(self.alphabet))]
        random.shuffle(random_matrix)
        random_matrix = np.asarray(random_matrix)
        # i know nothing about cryptography, here i will make something up
        self.encode_table = np.array((np.arange(len(self.alphabet)), random_matrix))
        self.decode_table = np.array((random_matrix, np.arange(len(self.alphabet))))
        
        # you could move this step into the encode/decode so that the key is 
        # actually required at those steps, but do it here for now so it's already calculated
        self.b = np.roll(self.decode_table[0:1,:][0], self.key)
        
        
    def int_encode(self, sequence):
        return [self.alph_to_idx[x] for x in sequence]
    def int_decode(self, sequence):
        return [self.idx_to_alph[x] for x in sequence]
    
    def encode(self, sequence):
        original_sequence = sequence
        
        # after this the sequence is int encoded
        sequence = self.int_encode(sequence)
        
        # after this the sequence is 'table' encoded
        cipher_encoded_sequence = []
        for c in sequence:
           col = np.argwhere(self.encode_table[0:1,:] == c)[0][1]
           cipher_encoded_sequence.append(self.encode_table[1,col])
        
        cipher_encoded_sequence = np.asarray(cipher_encoded_sequence)
        
        # after this the sequence is 'roll' encoded - do not exceed length of b! lol
        cipher_encoded_sequence = cipher_encoded_sequence + self.b[:len(cipher_encoded_sequence)]
        
        return original_sequence, cipher_encoded_sequence
    
    
    def decode(self, sequence):
        original_sequence = sequence
        
        # first remove the roll
        sequence = sequence - self.b[:len(sequence)]
        
        # then remove the table encoding
        cipher_decoded_sequence = []
        for c in sequence:
            col = np.argwhere(self.decode_table[0:1,:] == c)[0][1]
            cipher_decoded_sequence.append(self.decode_table[1,col])
        
        # then remove the int encoding - all decoded!
        cipher_decoded_sequence = self.int_decode(cipher_decoded_sequence)
        
        return original_sequence, cipher_decoded_sequence
    
        
    def summary(self):
        print('Length of alphabet: ', len(self.alphabet))
        print('Size of lookup table: ', self.encode_table.shape)
        print('Key: ', self.key)
        print(self.encode_table)

