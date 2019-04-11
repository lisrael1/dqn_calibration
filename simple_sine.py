from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from glob import glob
from keras.models import load_model


pd.set_option("display.max_columns",20) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)


class env():
    def __init__(self, original_offset= np.array([-1.1, 0.05, 0.11])):
        self.original_offset = original_offset
        self.offset = self.original_offset
        self.next_state=None
        self.ups=None
        self.number_of_actions=self.original_offset.shape[0]*2
        self.offset_history=pd.DataFrame([self.offset])
        self.reward_history=pd.DataFrame()
        self.perfect_sine=pd.DataFrame(np.linspace(0, 1, self.original_offset.shape[0]*33), columns=['time'])
        self.perfect_sine['sine']=np.sin(2 * np.pi * self.perfect_sine.time.values)
        # self.perfect_sine['sine']=np.zeros(self.perfect_sine.shape[0])
        self.mse=0
        self.current_sine, self.current_reward=self.generate_sine_with_offset(self.offset)

    def reset_env(self, new_offset):
        self.offset=new_offset
        self.current_sine, self.current_reward=self.generate_sine_with_offset(self.offset)

    def generate_sine_with_offset(self, offset):
        if 0:
            df = pd.DataFrame(np.linspace(0, 1, 99), columns=['time'])
            freq=np.random.uniform(1, 10)
            phase=np.random.uniform(1, 2*np.pi)
            freq=0
            phase=0
            df['sine'] = np.sin(freq*2 * np.pi * df.time.values+phase)
        else:
            df=self.perfect_sine.copy()
        offset = offset.tolist() * (df.shape[0] // len(offset))
        df['offset'] = offset
        df['sine_with_offsets'] = df.sine + df.offset
        new_mse = (df.sine_with_offsets - df.sine).pow(2).mean()
        if 0:
            walking_punishment=-0.05
            mse_threshold=0.1
            self.current_reward=0.3 if mse < self.mse else walking_punishment
            if mse<mse_threshold:
                self.current_reward = 1
        else:
            self.current_reward = (self.mse - new_mse)#*new_mse
            # self.current_reward = 1/new_mse
        self.mse = new_mse
        return df.sine_with_offsets.values, self.current_reward

    def get_reward_and_next_step_by_action(self, action_number):
        # mx=np.argmax(ups)
        place=action_number//2
        up_dn=-1 if action_number%2 else 1
        self.offset[place]+=up_dn*0.002
        self.offset_history=self.offset_history.append([self.offset])
        self.reward_history=self.reward_history.append([self.current_reward])
        self.current_sine, self.current_reward = self.generate_sine_with_offset(self.offset)
        return self.current_sine, self.current_reward  # those are the new values

    def random_action(self):
        return np.random.randint(self.number_of_actions)
        # actions=[0]*self.number_of_actions
        # actions[np.random.randint(0, self.number_of_actions)]=1
        # return actions

    def plot_history(self):
        import pylab as plt
        fig, ax = plt.subplots(5, 1)
        self.offset_history.reset_index(drop=True).plot(ax=ax[0], grid=True, title='current offset (need to be zeroes)')
        self.reward_history.reset_index(drop=True).plot(ax=ax[1], grid=True, title='reward history')
        pd.DataFrame(self.current_sine, index=self.perfect_sine.time).plot(ax=ax[2], grid=True, title='sine')  # title='current offset {}, original offset {}'.format(e.offset, e.original_offset))
        history = pd.read_csv('history_adc_output_estimator.csv', index_col=None)
        history.mean_squared_error.dropna().plot(ax=ax[3], title='training mse')
        history.val_mean_squared_error.dropna().plot(ax=ax[4], title='validation mse')
        # nn.action_values_history.reset_index(drop=True).plot(ax=ax[4], grid=True, title='action values history')
        # nn.action_values_history.idxmax(axis=1).reset_index(drop=True).plot(ax=ax[5], grid=True, title='chosen action history')
        plt.show()


class q:
    ''' should remmember single action - what we had, what we did and what happen next'''
    def __init__(self, memory_size):
        self.memory_size=memory_size
        self.memory = pd.DataFrame(columns=['old_sine', 'old_offsets', 'new_sine', 'action', 'reward'])

    def remember(self, old_sine, old_offsets, new_sine, action, reward):
        self.memory=self.memory.append(dict(old_sine=old_sine, old_offsets=old_offsets, new_sine=new_sine, action=action, reward=reward), ignore_index=True).reset_index(drop=True)
        if self.memory.shape[0]>self.memory_size:
            self.memory = self.memory.copy().sample(self.memory_size).reset_index(drop=True)

    def get_data(self, data_size):
        if self.memory.shape[0]<data_size:
            return self.memory.copy()
        return self.memory.copy().sample(data_size).reset_index(drop=True)


class model():
    def __init__(self, sine_samples, number_of_actions):
        self.nn=self.build_model(sine_samples, number_of_actions)
        self.action_values_history=pd.DataFrame()

    def build_model(self, sine_samples, number_of_actions, lr=0.001):
        if len(glob('model_checkpoint.h5')):
            model = load_model('model_checkpoint.h5')
        else:
            model = Sequential()
            model.add(Dense(sine_samples//8, input_dim=sine_samples))
            model.add(PReLU())
            # model.add(Dense(sine_samples*2, input_dim=sine_samples))
            # model.add(Dense(sine_samples*4))
            # model.add(PReLU())
            model.add(Dense(number_of_actions))
            model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        return model

    def generate_nn_data(self, memory):
        '''the q only remember 1 action, and we need here reward per each action, for all actions'''
        gamma_discount=0.2
        '''we need the next 2 for the nn training'''
        current_state = pd.DataFrame(memory.old_sine.apply(pd.Series).values)
        next_state = pd.DataFrame(memory.old_sine.apply(pd.Series).values)
        q_s = pd.DataFrame(self.nn.predict(current_state.values))
        learning = pd.concat([current_state, next_state, q_s], axis=1, keys=['current_state', 'next_state', 'q_s'])
        learning['memory_reward']=memory.reward
        learning['action']=memory.action.values
        learning['next_reward']=np.max(self.nn.predict(learning.next_state.values))
        learning['recursive_reward']=learning.memory_reward+gamma_discount*learning.next_reward
        for name, group in learning.groupby('action'):
            learning.loc[group.index.values, ('q_s', name)]=learning.recursive_reward
        # print(learning.drop(['current_state', 'next_state'], axis=1))

        if learning.shape[0] > 5:
            split_row = int(learning.shape[0] * 0.8)
            checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss',
                                                         verbose=0, save_best_only=False, save_weights_only=False,
                                                         mode='auto', period=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.6,
                                                          patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
            hist = CSVLogger('history_adc_output_estimator.csv', separator=',', append=True)
            self.nn.fit(learning.head(split_row).current_state.values, learning.head(split_row).q_s.values,
                        validation_data=(learning.head(-split_row).current_state.values, learning.head(-split_row).q_s.values),
                        # callbacks=[TQDMNotebookCallback(), reduce_lr, checkpoint, hist],
                        callbacks=[hist, checkpoint, reduce_lr],
                        epochs=8, batch_size=256, verbose=0)

        # TODO does the new reward should return to memory?
        return  # dict(validation_mse=h.history['val_mean_squared_error'], mse=h.history['mean_squared_error'])


dont_train=False
if dont_train and len(glob('model_checkpoint.h5')):
    print('model exist, loading it')
    nn=model(sine_samples=99, number_of_actions=6)
else:
    print('building and loading model')
    nn=model(sine_samples=99, number_of_actions=6)
    e=env()
    mem=q(2000000)
    rounds=4001 # for DC you need 2K, for single tone 4k and for multi tone

    def random_new_state_and_action(e):
        e=e[0]
        '''play one move and remember'''
        random_offset_for_random_state=np.random.normal(0, 0.5, 3)
        e.reset_env(random_offset_for_random_state)
        current_sine = e.current_sine
        epsilon = (np.random.uniform(0, 1) > 0.2) * i
        if epsilon:
            q_sa = np.argmax(nn.nn.predict(current_sine.reshape((1, -1)))[0])
        else:
            q_sa = e.random_action()
        new_sine, reward = e.get_reward_and_next_step_by_action(q_sa)
        return dict(old_sine=current_sine, old_offsets=random_offset_for_random_state, new_sine=new_sine, action=q_sa, reward=reward)


    for i in range(rounds) if 0 else tqdm(range(rounds)):
        one_play_for_memory = random_new_state_and_action([e])
        mem.remember(**one_play_for_memory)
        '''adapt nn to memory and update memory reward by recursia'''
        nn.generate_nn_data(mem.get_data(1000000))
        if i % 100 == 0 and i and 0:
            e.plot_history()

print('starting playing')
random_offset_for_random_state=np.random.normal(0, 0.9, 3)
play=env(random_offset_for_random_state)
nn_reward=[]
for _ in tqdm(range(2000)):
    action=nn.nn.predict(play.current_sine.reshape((1, -1)))
    nn_reward += [np.max(action)]
    play.get_reward_and_next_step_by_action(np.argmax(action))
pd.DataFrame(nn_reward).plot(title='what nn though the reward was')
play.plot_history()
