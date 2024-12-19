from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten,Input,GlobalAveragePooling1D
from keras.layers.pooling import AveragePooling1D
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
import keras

from tensorflow.keras.optimizers import SGD
from keras.models import Input, Model
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Layer,MultiHeadAttention,LayerNormalization,Embedding
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import random
random.seed(10)
np.random.seed(10)

#gpus=tf.config.experimental.list_physical_devices("GPU")
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu,True)
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 128 
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

#This code is inspired from :
# https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
vocab_size = 20000 #vocab for different patterns in reading
num_tokens_per_example = 99  #maxlength for attention
num_experts = 10
batch_size =512
num_tokens_per_batch = (batch_size * num_tokens_per_example)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ffn, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # The ffn can be either a standard feedforward network or a switch
        # layer with a Mixture of Experts.
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att'       : self.att,
            'ffn'       : self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb' : self.token_emb,
            'pos_emb' : self.pos_emb,
        })
        return config    

def create_feedforward_network(ff_dim, name=None):
    return Sequential(
        [Dense(ff_dim, activation="relu"), Dense(ff_dim)], name=name)


def load_balanced_loss(router_probs, expert_mask):
    num_experts = tf.shape(expert_mask)[-1]
    density = tf.reduce_mean(expert_mask, axis=0)
    density_proxy = tf.reduce_mean(router_probs, axis=0)
    loss = tf.reduce_mean(density_proxy * density) * tf.cast(
        (num_experts**2), tf.dtypes.float32)
    return loss

class Router(Layer):
    def __init__(self, num_experts, expert_capacity):
        self.num_experts = num_experts
        self.route = Dense(units=num_experts)
        self.expert_capacity = expert_capacity
        super(Router, self).__init__()

    def call(self, inputs, training=False):
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.
            router_logits += tf.random.uniform(
                shape=router_logits.shape, minval=0.9, maxval=1.1
            )
       
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        
        expert_gate, expert_index = tf.math.top_k(router_probs, k=1)
        
        expert_mask = tf.one_hot(expert_index, depth=self.num_experts)
        
        aux_loss = load_balanced_loss(router_probs, expert_mask)
        self.add_loss(aux_loss)
       
        position_in_expert = tf.cast(
            tf.math.cumsum(expert_mask, axis=0) * expert_mask, tf.dtypes.int32
        )
        # Keep only tokens that fit within expert capacity.
        expert_mask *= tf.cast(
            tf.math.less(
                tf.cast(position_in_expert, tf.dtypes.int32), self.expert_capacity
            ),
            tf.dtypes.float32,
        )
        expert_mask_flat = tf.reduce_sum(expert_mask, axis=-1)
        
        expert_gate *= expert_mask_flat
        
        combined_tensor = tf.expand_dims(
            expert_gate
            * expert_mask_flat
            * tf.squeeze(tf.one_hot(expert_index, depth=self.num_experts), 1),
            -1,
        ) * tf.squeeze(tf.one_hot(position_in_expert, depth=self.expert_capacity), 1)
       
        dispatch_tensor = tf.cast(combined_tensor, tf.dtypes.float32)

        return dispatch_tensor, combined_tensor

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_experts' : self.num_experts,
            'route' : self.route,
            'expert_capacity' : self.expert_capacity,
        })
        return config 


class Switch(Layer):
    def __init__(self, num_experts, embed_dim, num_tokens_per_batch, capacity_factor=1):
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.experts = [
            create_feedforward_network(embed_dim) for _ in range(num_experts)
        ]

        self.expert_capacity = num_tokens_per_batch // self.num_experts
        self.router = Router(self.num_experts, self.expert_capacity)
        super(Switch, self).__init__()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_tokens_per_example = tf.shape(inputs)[1]

        # inputs shape: [num_tokens_per_batch, embed_dim]
        inputs = tf.reshape(inputs, [num_tokens_per_batch, self.embed_dim])
        # dispatch_tensor shape: [expert_capacity, num_experts, tokens_per_batch]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        # expert_inputs shape: [num_experts, expert_capacity, embed_dim]
        expert_inputs = tf.einsum("ab,acd->cdb", inputs, dispatch_tensor)
        expert_inputs = tf.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = tf.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = tf.stack(expert_output_list, axis=1)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = tf.einsum(
            "abc,xba->xc", expert_outputs, combine_tensor
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]
        outputs = tf.reshape(
            expert_outputs_combined,
            [batch_size, num_tokens_per_example, self.embed_dim],
        )
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_experts' : self.num_experts,
            'embed_dim' : self.embed_dim,
            'router' : self.router,
            'experts' : self.experts,
            'expert_capacity' : self.expert_capacity,
        })
        return config 

class LPpool(Layer):
    def __init__(self, pool_size, strides=None, padding='same'):
        super(LPpool,self).__init__()
        self.avgpool=tf.keras.layers.AveragePooling1D(pool_size,strides,padding)

    def call(self, x):
        x = tf.math.pow(tf.math.abs(x), 2)
        x = self.avgpool(x)
        x = tf.math.pow(x, 1.0 / 2)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'avgpool'       : self.avgpool,
        })
        return config

class Switch_Tran(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "Switch_Tran"
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,**load_kwargs):

        print("...............Switch Transformer partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

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
                    filepath = 'St-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    model.fit(train_x,train_y,validation_data=(v_x,v_y),epochs=self.n_epochs,callbacks=[checkpoint],batch_size=self.batch_size)
                    model.load_weights(filepath)

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

    def return_network(self):
        '''Creates the Switch Trans module
        '''
        from keras.callbacks import EarlyStopping
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        vocab_size = 20000 #vocab for different patterns in reading
        num_tokens_per_example = self.sequence_length  #maxlength for attention
        num_experts = 10
        batch_size =50
        num_tokens_per_batch = (batch_size * num_tokens_per_example)

        with strategy.scope():
        	switch = Switch(num_experts, embed_dim, num_tokens_per_batch)
        	transformer_block = TransformerBlock(ff_dim, num_heads, switch)

        	inputs = Input(shape=(num_tokens_per_example, 1))
        	embedding_layer = TokenAndPositionEmbedding(num_tokens_per_example, vocab_size, embed_dim)
        	x = Conv1D(16,4,activation="linear",padding="same",strides=1)(inputs)
        	x = LPpool(pool_size=2)(x)
        	x = embedding_layer(x)
        	x = transformer_block(x)
        	x = Flatten()(x)
        	x = Dropout(0.1)(x)
        	x = Dense(num_tokens_per_example, activation="relu")(x)
        	outputs = Dropout(0.1)(x)
        	model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        early_stopping = EarlyStopping(monitor='loss', verbose=1, mode='min')
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
