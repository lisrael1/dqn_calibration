from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from glob import glob
from keras.models import load_model


pd.set_option("display.max_columns",20) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)


class Env():
    def __init__(self, v_offset_en = True, t_offset_en = True, num_SARs=3, SAR_samples=33):
        self.num_SARs = num_SARs
        self.num_SAR_samples = SAR_samples
        self.v_offset_en = v_offset_en
        self.t_offset_en = t_offset_en
        self.v_offset = np.random.uniform(-0.5, 0.5, 3)*v_offset_en
        self.t_offset = np.random.uniform(-0.025, 0.025, 3)*t_offset_en
        self.number_of_actions=self.num_SARs*2*(int(v_offset_en)+int(t_offset_en))
        # self.offset_history=pd.DataFrame([self.offset])
        # self.reward_history=pd.DataFrame()
        self.perfect_sine=pd.DataFrame(np.linspace(0, 1, self.num_SARs*self.num_SAR_samples), columns=['time'])
        self.perfect_sine['value']=np.sin(2 * np.pi * self.perfect_sine.time.values)
        self.mse=0
        self.current_sine, self.current_reward=self.generate_noisy_sine()

    def reset_env(self):
        self.v_offset = np.random.uniform(-0.5, 0.5, 3)*self.v_offset_en
        self.t_offset = np.random.uniform(-0.025, 0.025, 3)*self.t_offset_en
        self.current_sine, self.current_reward=self.generate_noisy_sine()

    def generate_noisy_sine(self):
        time_vals = self.perfect_sine.time.values
        time_offsets = np.asarray(list(self.t_offset) * self.num_SAR_samples)
        sine_with_time_offsets = np.sin(2 * np.pi * (time_vals+time_offsets))
        v_offset = np.asarray(list(self.v_offset) * self.num_SAR_samples)
        noisy_sine = sine_with_time_offsets + v_offset
        # plt.plot(np.arange(0,99,3), noisy_sine[::3])
        # plt.plot(np.arange(1,99,3), noisy_sine[1::3])
        # plt.plot(np.arange(2,99,3), noisy_sine[2::3])
        # plt.plot(noisy_sine,'b')
        # plt.plot(self.perfect_sine.value.values,'g')
        # plt.grid()
        # plt.show()
        # phase=np.random.uniform(1, 2*np.pi)
        new_mse = np.mean((noisy_sine - self.perfect_sine.value.values)**2)
        reward = np.clip(1000*((self.mse - new_mse)/self.mse), -1, 1)
        # reward = np.sign((self.mse - new_mse)/self.mse)
        self.mse = new_mse
        return noisy_sine, reward

    def get_reward_and_next_state_by_action(self, action_number):
        # Actions: SAR1: v_up, v_dn, t_up, t_dn, SAR2:...
        SAR=int(action_number//(self.number_of_actions/self.num_SARs))
        if (self.v_offset_en and not self.t_offset_en): action_in_sar={0:'v_up',1:'v_dn'}[action_number % 2]
        elif (not self.v_offset_en and self.t_offset_en): action_in_sar = {0: 't_up', 1: 't_dn'}[action_number % 2]
        else: action_in_sar={0:'v_up',1:'v_dn',2:'t_up',3:'t_dn'}[action_number % 4]
        if action_in_sar.startswith("v"):
            if action_in_sar.endswith("up"):
                self.v_offset[SAR] += 0.02
            else: self.v_offset[SAR] -= 0.02
        else:
            if action_in_sar.endswith("up"):
                self.t_offset[SAR] += 0.0025
            else: self.t_offset[SAR] -= 0.0025

        self.v_offset = np.clip(self.v_offset, -0.5, 0.5)
        # self.v_offset = np.clip(self.v_offset, 0, 0)
        self.t_offset = np.clip(self.t_offset, -0.025, 0.025)
        # self.t_offset = np.clip(self.t_offset, 0, 0)

        self.current_sine, self.current_reward = self.generate_noisy_sine()
        # self.offset_history=self.offset_history.append([self.offset])
        # self.reward_history=self.reward_history.append([self.current_reward])
        return self.current_sine, self.current_reward


    def random_action(self):
        return np.random.randint(self.number_of_actions)


    # def plot_history(self):
    #     import pylab as plt
    #     fig, ax = plt.subplots(5, 1)
    #     self.offset_history.reset_index(drop=True).plot(ax=ax[0], grid=True, title='current offset (need to be zeroes)')
    #     self.reward_history.reset_index(drop=True).plot(ax=ax[1], grid=True, title='reward history')
    #     pd.DataFrame(self.current_sine, index=self.perfect_sine.time).plot(ax=ax[2], grid=True, title='sine')  # title='current offset {}, original offset {}'.format(e.offset, e.original_offset))
    #     history = pd.read_csv('history_adc_output_estimator.csv', index_col=None)
    #     history.mean_squared_error.dropna().plot(ax=ax[3], title='training mse')
    #     history.val_mean_squared_error.dropna().plot(ax=ax[4], title='validation mse')
    #     # nn.action_values_history.reset_index(drop=True).plot(ax=ax[4], grid=True, title='action values history')
    #     # nn.action_values_history.idxmax(axis=1).reset_index(drop=True).plot(ax=ax[5], grid=True, title='chosen action history')
    #     plt.show()


class Memory:
    def __init__(self, memory_size):
        self.memory_size=memory_size
        self.memory = pd.DataFrame(columns=['old_sine', 'q_s'])

    def remember(self, old_sine, q_s):
        self.memory=self.memory.append(dict(old_sine=old_sine, q_s=q_s), ignore_index=True).reset_index(drop=True)
        if self.memory.shape[0]>self.memory_size:
            self.memory = self.memory.copy().tail(self.memory_size).reset_index(drop=True)

    def get_data(self, data_size):
        if self.memory.shape[0]<data_size:
            return self.memory.copy()
        return self.memory.copy().sample(data_size).reset_index(drop=True)


class Model():
    def __init__(self, sine_samples, number_of_actions):
        self.nn=self.build_model(sine_samples, number_of_actions)
        # self.action_values_history=pd.DataFrame()

    def build_model(self, sine_samples, number_of_actions, lr=0.001):
        if len(glob('model_checkpoint.h5')):
            model = load_model('model_checkpoint.h5')
        else:
            model = Sequential()
            model.add(Dense(sine_samples, input_dim=sine_samples, activation='tanh'))
            # model.add(Dense(sine_samples, activation='tanh'))
            model.add(Dense(sine_samples//2, activation='tanh'))
            model.add(Dense(sine_samples // 2, activation='tanh'))
            model.add(Dense(sine_samples // 4, activation='tanh'))
            model.add(Dense(sine_samples//8, activation='tanh'))
            model.add(Dense(number_of_actions, activation=None))
            model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        return model

    def generate_nn_data(self, memory):
        current_state = memory.old_sine.apply(pd.Series).values
        q_s = memory.q_s.apply(pd.Series).values
        data = (current_state, q_s)
        return data


    def train_model(self, X, Y):
        reduce_lr = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.6, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        hist = CSVLogger('history_adc_output_estimator.csv', separator=',', append=True)
        self.nn.fit(X, Y, batch_size=10, epochs=1, verbose=0, callbacks=[reduce_lr, hist])
        return



if __name__ == '__main__':

    dont_train=False
    if dont_train and len(glob('model_checkpoint.h5')):
        print('model exist, loading it')
        model = Model(sine_samples=None, number_of_actions=None)
    else:
        adc = Env()
        model = Model(sine_samples=adc.num_SARs*adc.num_SAR_samples, number_of_actions=adc.number_of_actions)
        mem=Memory(memory_size=500)
        # rounds=200000 # for DC you need 2K, for single tone 4k and for multi tone
        epsilon = 0.7
        avg_reward = 0
        count = 0
        i=0
        v_os_hist = [[], [], []]
        t_os_hist = [[],[],[]]
        while(True):
            if (i+1)%20000 == 0:
                adc.reset_env()
                print("New sample")
            current_sine = adc.current_sine
            if (i+1) % 40000 == 0:
                epsilon = epsilon / 2
                print("epsilon="+str(epsilon))
            q_s = model.nn.predict(current_sine.reshape((1, -1)))[0]
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q_s)
            else:
                action = adc.random_action()
            new_sine, reward = adc.get_reward_and_next_state_by_action(action)
            avg_reward += reward
            q_s_next = model.nn.predict(new_sine.reshape((1, -1)))[0]
            q_s[action] = reward + 0.5*np.amax(q_s_next)
            # if reward/np.amax(q_s_next) < 0.01: count+=1
            # plt.plot(new_sine-current_sine); plt.show()
            mem.remember(current_sine, q_s)
            if (i+1) % 100 == 0:
                X, Y = model.generate_nn_data(mem.get_data(100))
                model.train_model(X, Y)
            if i % 1000 == 0:
                print("round {i}, rmse={mse}\n\tv offset = {v_ofst}\n\tt offset={t_ofst}\n\taverage reward={avg}\n".format(i=i, mse=np.sqrt(adc.mse), v_ofst=(adc.v_offset/0.02).astype(int), t_ofst=(adc.t_offset/0.0025).astype(int), avg=avg_reward/1000))
                avg_reward = 0

            i+=1
            if i==200000: break
            # if i % 100 == 0 and i and 0:
            #     e.plot_history()

    # print('starting playing')
    # random_offset_for_random_state=np.random.normal(0, 0.9, 3)
    # play=env(random_offset_for_random_state)
    # nn_reward=[]
    # for _ in range(2000):
    #     action=nn.nn.predict(play.current_sine.reshape((1, -1)))
    #     nn_reward += [np.max(action)]
    #     play.get_reward_and_next_step_by_action(np.argmax(action))
    # pd.DataFrame(nn_reward).plot(title='what nn though the reward was')
    # play.plot_history()
