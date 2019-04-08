from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.callbacks import CSVLogger


class env():
    def __init__(self):
        self.original_offset = np.array([-1.1, 0.05, 0.11])
        self.offset = self.original_offset
        self.next_state=None
        self.ups=None
        self.number_of_actions=6
        self.offset_history=pd.DataFrame([self.offset])
        self.reward_history=pd.DataFrame()
        self.perfect_sine=pd.DataFrame(np.linspace(0, 1, self.original_offset.shape[0]*33), columns=['time'])
        self.perfect_sine['sine']=np.sin(2 * np.pi * self.perfect_sine.time.values)
        self.mse=0
        self.current_sine, self.current_reward=self.generate_sine_with_offset(self.offset)

    def generate_sine_with_offset(self, offset):
        if 0:
            df = pd.DataFrame(np.linspace(0, 1, 99), columns=['time'])
            df['sine'] = np.sin(2 * np.pi * df.time.values)
        else:
            df=self.perfect_sine.copy()
        offset = offset.tolist() * (df.shape[0] // len(offset))
        df['offset'] = offset
        df['sine_with_offsets'] = df.sine + df.offset
        mse = (df.sine_with_offsets - df.sine).pow(2).mean()
        if 0:
            walking_punishment=-0.05
            mse_threshold=0.1
            self.current_reward=0.3 if mse < self.mse else walking_punishment
            if mse<mse_threshold:
                self.current_reward = 1
        else:
            self.current_reward = mse - self.mse
        self.mse = mse
        return df.sine_with_offsets.values, self.current_reward

    def update_ups_and_state(self, ups):
        mx=np.argmax(ups)
        place=mx//2
        up_dn=-1 if mx%2 else 1
        self.offset[place]+=up_dn*0.02
        self.offset_history=self.offset_history.append([self.offset])
        self.reward_history=self.reward_history.append([self.current_reward])
        self.current_sine, self.current_reward = self.generate_sine_with_offset(self.offset)
        return self.current_sine, self.current_reward  # those are the new values

    def random_action(self):
        actions=[0]*self.number_of_actions
        actions[np.random.randint(0, self.number_of_actions)]=1
        return actions


class q:
    ''' should remmember single action - what we had, what we did and what happen next'''
    def __init__(self, memory_size):
        self.memory_size=memory_size
        self.memory = pd.DataFrame(columns=['old_sine', 'new_sine', 'action', 'reward'])

    def remember(self, old_sine, new_sine, action, reward):
        self.memory=self.memory.append(dict(old_sine=old_sine, new_sine=new_sine, action=action, reward=reward), ignore_index=True).reset_index(drop=True)
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
        model = Sequential()
        model.add(Dense(sine_samples*2, input_dim=sine_samples))
        model.add(PReLU())
        model.add(Dense(sine_samples*4))
        model.add(PReLU())
        model.add(Dense(sine_samples))
        model.add(PReLU())
        model.add(Dense(number_of_actions))
        model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        return model

    def generate_nn_data(self, memory):
        '''the q only remember 1 action, and we need here reward per each action, for all actions'''
        discount=0.2
        '''we need the next 2 for the nn training'''
        inputs = pd.DataFrame(memory.old_sine.apply(pd.Series).values)
        outputs = pd.DataFrame(memory.action.apply(pd.Series).values)
        learning = pd.concat([inputs, outputs], axis=1, keys=['inputs', 'outputs'])

        for idx, mem in memory.iterrows():
            actions_by_memory=mem.action
            next_sine_record_by_memory=mem.new_sine.reshape((1, -1))
            next_size_actions_by_nn=self.nn.predict(next_sine_record_by_memory)[0]
            self.action_values_history=self.action_values_history.append([next_size_actions_by_nn])
            action_predict_for_next_state=np.argmax(next_size_actions_by_nn)
            learning.loc[idx, ('outputs', np.argmax(actions_by_memory))] = mem.reward if mem.reward == 1 else mem.reward + discount*action_predict_for_next_state-0.2
        if memory.shape[0]>10:
            split_row=int(learning.shape[0]*0.8)
            hist = CSVLogger('history_adc_output_estimator.csv', separator=',', append=True)
            self.nn.fit(learning.head(split_row).inputs.values, learning.head(split_row).outputs.values,
                            validation_data=(learning.head(-split_row).inputs.values, learning.head(-split_row).outputs.values),
                            # callbacks=[TQDMNotebookCallback(), reduce_lr, checkpoint, hist],
                            callbacks=[hist],
                            epochs=8, batch_size=16, verbose=1)


nn=model(sine_samples=99, number_of_actions=6)
e=env()
mem=q(100)
rounds=1000
for i in range(rounds) if 1 else tqdm(range(rounds)):
    '''play one move and remember'''
    current_sine=e.current_sine
    epsilon=(np.random.uniform(0, 1) > 0.2)*i
    q_s=nn.nn.predict(current_sine.reshape((1, -1)))[0].tolist() if epsilon else e.random_action()
    new_sine, reward = e.update_ups_and_state(q_s)
    mem.remember(old_sine=current_sine, new_sine=new_sine, action=q_s, reward=reward)
    '''adapt nn to memory and update memory reward by recursia'''
    nn.generate_nn_data(mem.get_data(10))
    if i % 100==0 and i:
        fig, ax = plt.subplots(6, 1)
        e.offset_history.reset_index(drop=True).plot(ax=ax[0], grid=True, title='current offset (need to be zeroes)')
        e.reward_history.reset_index(drop=True).plot(ax=ax[1], grid=True, title='reward history')
        pd.DataFrame(e.current_sine, index=e.perfect_sine.time).plot(ax=ax[2], grid=True, title='sine')  # title='current offset {}, original offset {}'.format(e.offset, e.original_offset))
        history = pd.read_csv('history_adc_output_estimator.csv', index_col=None)
        history.val_mean_squared_error.plot(ax=ax[3], title='validation mse')
        nn.action_values_history.reset_index(drop=True).plot(ax=ax[4], grid=True, title='action values history')
        nn.action_values_history.idxmax(axis=1).reset_index(drop=True).plot(ax=ax[5], grid=True, title='chosen action history')
        plt.show()

