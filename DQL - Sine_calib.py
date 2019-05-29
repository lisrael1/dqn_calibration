from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten
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

V_STEP = 0.02
T_STEP = 0.0025


class Env():
    def __init__(self, v_offset_en = True, t_offset_en = True, num_SARs=3, SAR_samples=33):
        self.num_SARs = num_SARs
        self.num_SAR_samples = SAR_samples
        self.v_offset_en = v_offset_en
        self.t_offset_en = t_offset_en
        self.v_offset = np.random.randint(-25, 25, 3)*v_offset_en
        self.t_offset = np.random.randint(-10, 10, 3)*t_offset_en
        self.number_of_actions=self.num_SARs*2*(int(v_offset_en)+int(t_offset_en))
        self.perfect_sine=pd.DataFrame(np.linspace(0, 1, self.num_SARs*self.num_SAR_samples), columns=['time'])
        self.mse=0
        self.current_sine, self.current_reward=self.generate_noisy_sine()

    def reset_env(self):
        self.v_offset = np.random.randint(-25, 25, 3)*self.v_offset_en
        self.t_offset = np.random.randint(-10, 10, 3)*self.t_offset_en
        self.current_sine, self.current_reward=self.generate_noisy_sine()

    def generate_noisy_sine(self):
        phase = np.random.uniform(0, 2*np.pi)*0
        time_vals = self.perfect_sine.time.values
        self.perfect_sine['value'] = np.sin(2 * np.pi * self.perfect_sine.time.values + phase)
        time_offsets = np.asarray(list(self.t_offset*T_STEP) * self.num_SAR_samples)
        sine_with_time_offsets = np.sin(2 * np.pi * (time_vals+time_offsets) + phase)
        v_offset = np.asarray(list(self.v_offset*V_STEP) * self.num_SAR_samples)
        noisy_sine = sine_with_time_offsets + v_offset
        # plt.plot(np.arange(0,99,3), noisy_sine[::3])
        # plt.plot(np.arange(1,99,3), noisy_sine[1::3])
        # plt.plot(np.arange(2,99,3), noisy_sine[2::3])
        # plt.plot(noisy_sine,'b')
        # plt.plot(self.perfect_sine.value.values,'g')
        # plt.grid()
        # plt.show()
        new_mse = np.mean((noisy_sine - self.perfect_sine.value.values)**2)
        reward = np.clip(1000*((self.mse - new_mse)/self.mse), -1, 1)
        self.mse = new_mse
        self.generate_codes_df()
        return noisy_sine, reward

    def get_reward_and_next_state_by_action(self, action_number):
        # Actions: SAR1: v_up, v_dn, t_up, t_dn, SAR2:...
        SAR=int(action_number//(self.number_of_actions/self.num_SARs))
        if (self.v_offset_en and not self.t_offset_en): action_in_sar={0:'v_up',1:'v_dn'}[action_number % 2]
        elif (not self.v_offset_en and self.t_offset_en): action_in_sar = {0: 't_up', 1: 't_dn'}[action_number % 2]
        else: action_in_sar={0:'v_up',1:'v_dn',2:'t_up',3:'t_dn'}[action_number % 4]
        if action_in_sar.startswith("v"):
            if action_in_sar.endswith("up"):
                self.v_offset[SAR] += 1
            else: self.v_offset[SAR] -= 1
        else:
            if action_in_sar.endswith("up"):
                self.t_offset[SAR] += 1
            else: self.t_offset[SAR] -= 1

        self.v_offset = np.clip(self.v_offset, -25, 25)
        self.t_offset = np.clip(self.t_offset, -10, 10)

        self.current_sine, self.current_reward = self.generate_noisy_sine()
        return self.current_sine, self.current_reward


    def random_action(self):
        return np.random.randint(self.number_of_actions)

    def generate_codes_df(self):
        self.codes_df = pd.DataFrame()
        self.codes_df["v_offset"] = self.v_offset
        self.codes_df["t_offset"] = self.t_offset
        self.codes_df.index.name = "SAR#"
        return


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

    def build_model(self, sine_samples, number_of_actions, lr=0.001):
        if len(glob('model_checkpoint.h5')):
            model = load_model('model_checkpoint.h5')
        else:
            model = Sequential()
            # # Fully connected
            # model.add(Dense(sine_samples, input_dim=sine_samples, activation='tanh'))
            # model.add(Dense(sine_samples//2, activation='tanh'))
            # model.add(Dense(sine_samples // 2, activation='tanh'))
            # model.add(Dense(sine_samples // 4, activation='tanh'))
            # model.add(Dense(sine_samples//8, activation='tanh'))

            # Conv net
            model.add(Conv1D(64, kernel_size=9, activation='tanh', input_shape=(sine_samples,1)))
            model.add(Conv1D(16, kernel_size=9, activation='tanh'))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(4, kernel_size=9, activation='tanh'))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(1, kernel_size=9, activation='tanh'))
            model.add(Flatten())
            model.add(Dense(50, activation='tanh'))
            model.add(Dense(25, activation='tanh'))
            model.add(Dense(number_of_actions, activation=None))
            model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        return model

    def generate_nn_data(self, memory):
        current_state = memory.old_sine.apply(pd.Series).values
        current_state = np.expand_dims(current_state, axis=2)
        q_s = memory.q_s.apply(pd.Series).values
        data = (current_state, q_s)
        return data


    def train_model(self, X, Y):
        reduce_lr = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.6, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        hist = CSVLogger('history_adc_output_estimator.csv', separator=',', append=True)
        self.nn.fit(X, Y, batch_size=10, epochs=3, verbose=0, callbacks=[reduce_lr, hist])
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
        max_rounds = 200000
        epsilon = 0.7
        avg_reward = 0
        i=0
        v_os_hist = [[], [], []]
        t_os_hist = [[],[],[]]
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10, 6))
        plt.ion()
        plt.show()
        sample_num = 1
        while(True):
            if (i+1)%10000 == 0:
                adc.reset_env()
                print("New sample")
                sample_num += 1
            current_sine = adc.current_sine
            if (i+1) % 10000 == 0:
                epsilon = epsilon / 2
                print("epsilon="+str(epsilon))
            q_s = model.nn.predict(current_sine.reshape((1, -1, 1)))[0]
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q_s)
            else:
                action = adc.random_action()
            new_sine, reward = adc.get_reward_and_next_state_by_action(action)
            avg_reward += reward
            q_s_next = model.nn.predict(new_sine.reshape((1, -1, 1)))[0]
            q_s[action] = reward + 0.5*np.amax(q_s_next)
            # plt.plot(new_sine-current_sine); plt.show()
            mem.remember(current_sine, q_s)
            if (i+1) % 100 == 0:
                X, Y = model.generate_nn_data(mem.get_data(100))
                model.train_model(X, Y)
            if i % 100 == 0:
                print("round {i}, rmse={mse}\n\tv offset = {v_ofst}\n\tt offset={t_ofst}\n\taverage reward={avg}\n".format(i=i, mse=np.sqrt(adc.mse), v_ofst=(adc.v_offset).astype(int), t_ofst=(adc.t_offset).astype(int), avg=avg_reward/100))
                ax1.clear(); ax2.clear(); ax3.clear()
                ax1.axis([0, 1, -25, 25])
                adc.codes_df.T.plot.bar(rot=0, ax=ax1)
                ax1.grid(True)
                ax2.plot(adc.current_sine)
                state = 'Round = {round}\nSample Number = {sample_num}\nRMSE[mv] = {rmse}\nAverage Reward = {reward:{p}}\n'.format(p='3f',round=i, sample_num=sample_num, reward=avg_reward/100, rmse=1e3 * np.sqrt(adc.mse))
                ax3.text(x=0.2, y=0.25, s=state, family='monospace')
                plt.draw()  # or plt.show(block=False)
                plt.pause(0.0001)
                avg_reward = 0
            i += 1
            if i == max_rounds: break

