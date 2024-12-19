from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten, Multiply, GlobalAveragePooling1D
from keras.layers import Dense, Conv2D, Flatten, Reshape, Lambda, dot, Activation, concatenate, Conv1D, SpatialDropout1D, BatchNormalization, add  

import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from tcn import TCN
import keras
from tensorflow.keras.backend import *
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.models import Input, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
import keras.backend as K
import random
random.seed(10)
np.random.seed(10)


import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 128 
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync




class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass


class TempConv(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "Temporal Conv"
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size',512)
        self.nb_filters = params.get('nb_filters',16)
        self.filter_length = params.get('filter_length',3)
        self.dilations = params.get('dilations',[1,2,4,8,16,32,64,128])
        self.dropout = params.get('dorpout',0.3)
        self.appliance_params = params.get('appliance_params',{})
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,**load_kwargs):

        print("...............TCN partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print (len(train_main))
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs,axis=0)
            app_df_values = app_df.values.reshape((-1,self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    #filepath = 'bitcn-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    #checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./bitemp")
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    model.fit(train_x,train_y,validation_data=(v_x,v_y),epochs=self.n_epochs,callbacks=[tensorboard_callback],batch_size=self.batch_size)
                    #model.load_weights(filepath)

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:

            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))

            for appliance in self.models:

                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array ,batch_size=self.batch_size)

                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured
                
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                #################
                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

    def residual_block(self, x, dilation_rate, nb_filters, kernel_size, padding, activation='relu', dropout_rate=0, kernel_initializer='he_normal'):
                   prev_x = x
                   for k in range(2):
                   	x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, kernel_initializer=kernel_initializer, padding=padding)(x)
                   	x = BatchNormalization()(x)
                   	x = Activation('relu')(x)
                   	#x = SpatialDropout1D(rate=dropout_rate)(inputs=x)
                   	
                   	# 1x1 conv to match the shapes (channel dimension).
                   	prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
                   	res_x = add([prev_x, x])
                   	#####################################
                   	res_x = Activation(activation)(res_x)
                   	return res_x, x
                   	
    
    
    def return_network(self):
        from keras.callbacks import EarlyStopping
        with strategy.scope():
        	inp = Input(batch_shape=(None, self.sequence_length, 1), name="input")
        	x = Conv1D(filters=self.nb_filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(inp)
        	skip_connections = []
        	for d in self.dilations:
        	    x, skip_out = self.residual_block(x, dilation_rate=d,nb_filters=self.nb_filters,kernel_size=self.filter_length, padding = 'same', activation = 'relu', dropout_rate=self.dropout)
        	    skip_connections.append(skip_out)
        	x = add(skip_connections)
        	
        	d_out = Dense(1, activation='linear', name='output')(x)
        	model = Model(inputs=inp, outputs=d_out)
        	
        	
        	
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        early_stopping = EarlyStopping(monitor='loss', verbose=1, patience=3)
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            #new_mains = pd.DataFrame(new_mains)
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()


                processed_app_dfs = []
                for app_df in app_df_lst:                    
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                    new_app_readings = (new_app_readings - app_mean) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                    
                    
                appliance_list.append((app_name, processed_app_dfs))
                #new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
                #print (new_mains.shape, new_app_readings.shape, app_name)

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
