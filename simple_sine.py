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

pd.set_option("display.max_columns",100) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)


class env():
    '''
        what env should do? what we us it for?
        generate new impairments with new sine
        make a move by choosing action and updating sine and reward
    '''
    def __init__(self):
        self.number_of_sars=3
        self.operations_factors=dict(volt_offset=1e-1, gain_offset=0, time_offset=0)
        self.impairments = pd.Series(index=pd.MultiIndex.from_product([self.operations_factors.keys(), range(self.number_of_sars)])).fillna(0)  # , list('ud')
        self.all_actions = pd.Series(index=pd.MultiIndex.from_product([self.operations_factors.keys(), range(self.number_of_sars), ['up','dn']])).fillna(0)  # , list('ud')
        self.impairments_in_codes = self.impairments.copy()
        self.number_of_samples_per_sar=25

        # self.impairments_history=pd.DataFrame(columns=self.impairments.index).copy()
        # self.reward_history=pd.DataFrame()

        self.sine_freq_hz=1
        self.sine_phase_rad=0
        self.initial_random_code = 10
        self.reward_clipped_value = 10000
        self.reward_factor = 10000

        self.mse=0
        self._update_impairments()
        self.current_sine, self.current_reward=self._generate_sine_by_impairments()

    def get_all_actions_rewards_as_table(self, codes_list):
        table=self.all_actions.copy()
        table.iloc[:]=codes_list
        return table

    def _cast_code_list_to_impairment_values(self, codes):
        '''
            casting the codes to voltage/gain etc. values
        :param codes: list of integers, one hot, with up and down. if you give non one hot, it will translate them all to voltage time etc. values
        :return:
        '''
        values=self.impairments.copy()
        codes=np.array(codes)
        values.iloc[:] = codes[::2]-codes[1::2]
        for impairment, val in self.operations_factors.items():
            values.loc[values.index.get_level_values(0) == impairment] *= val
        return values

    def _update_impairments(self, impairments_update_code_vector=None):
        if impairments_update_code_vector is None:
            self.impairments *= 0
            self.impairments_in_codes *= 0
            impairments_update_code_vector = np.random.randint(-self.initial_random_code, self.initial_random_code, self.impairments.shape[0]*2)
            # self.sine_phase_rad=np.random.uniform(0, 2*np.pi)
        self.impairments_in_codes+=impairments_update_code_vector[::2]-impairments_update_code_vector[1::2]
        self.impairments += self._cast_code_list_to_impairment_values(impairments_update_code_vector)

    def reset_env(self):
        self._update_impairments()
        self.current_sine, self.current_reward=self._generate_sine_by_impairments()

    def _generate_sine_by_impairments(self):
        df=pd.DataFrame()
        df['time'] = np.linspace(0, 1, self.number_of_samples_per_sar * self.number_of_sars)
        df['perfect_sine']=np.sin(2*np.pi*self.sine_freq_hz*df.time+self.sine_phase_rad)
        df['time_offset'] = self.impairments.loc[self.impairments.index.get_level_values(0) == 'time_offset'].values.tolist()*self.number_of_samples_per_sar
        df['volt_offset'] = self.impairments.loc[self.impairments.index.get_level_values(0) == 'volt_offset'].values.tolist()*self.number_of_samples_per_sar
        df['gain_offset'] = self.impairments.loc[self.impairments.index.get_level_values(0) == 'gain_offset'].values.tolist()*self.number_of_samples_per_sar
        df['time_with_offset']=df.time+df.time_offset
        df['sine_with_time_offset']=np.sin(2*np.pi*self.sine_freq_hz*df.time_with_offset+self.sine_phase_rad)
        df['sine_with_time_and_gain_offset']=np.multiply(df.sine_with_time_offset.values, 1-df.gain_offset.values)
        df['sine_with_volt_gain_and_time_offset']=df.sine_with_time_and_gain_offset+df.volt_offset
        new_mse = (df.sine_with_volt_gain_and_time_offset - df.perfect_sine).pow(2).mean()
        if 0:
            walking_punishment=-0.05
            mse_threshold=0.1
            self.current_reward=0.3 if mse < self.mse else walking_punishment
            if mse<mse_threshold:
                self.current_reward = 1
        else:
            self.current_reward = self.reward_factor*(self.mse - new_mse)#/self.mse
            # self.current_reward = 1/new_mse
            # print(self.current_reward)
            self.current_reward=np.clip(self.current_reward, -self.reward_clipped_value, self.reward_clipped_value)
        self.mse = new_mse
        return df.sine_with_volt_gain_and_time_offset.values, self.current_reward

    def get_action_and_update_current_reward_and_next_step(self, action_number):
        code_list=np.zeros(self.impairments.shape[0]*2)
        code_list[action_number]=1
        self._update_impairments(code_list)
        self.current_sine, self.current_reward = self._generate_sine_by_impairments()
        return self.current_sine, self.current_reward

    def random_action_index(self):
        return np.random.randint(self.impairments.shape[0])


class q:
    ''' should remmember single action - what we had, what we did and what happen next'''
    def __init__(self, memory_size):
        self.memory_size=memory_size
        self.memory = None

    def remember(self, sine, actions_rewards):
        if self.memory is None:
            sine_col = pd.DataFrame(list(range(len(sine))), columns=['inx'])
            sine_col['typ'] = 'sine'
            action_col = pd.DataFrame(list(range(len(actions_rewards))), columns=['inx'])
            action_col['typ'] = 'actions_rewards'
            cols=pd.concat([sine_col.T, action_col.T], axis=1)[::-1]
            self.memory = pd.DataFrame(columns=pd.MultiIndex.from_frame(cols.T))
        new_data=pd.DataFrame(np.hstack([sine, actions_rewards]).reshape((1, -1)), columns=self.memory.columns)
        self.memory = pd.concat([self.memory, new_data]).reset_index(drop=True)
        if self.memory.shape[0]>self.memory_size:
            self.memory = self.memory.copy().sample(self.memory_size).reset_index(drop=True)

    def get_data(self, data_size):
        data_size = np.min([data_size, self.memory.shape[0]])
        return self.memory.copy().sample(data_size).reset_index(drop=True)


class model():
    def __init__(self, sine_samples, number_of_actions):
        self.action_values_history=pd.DataFrame()
        self.lr = 1e-2
        self.nn=self.build_model(sine_samples, number_of_actions)

    def build_model(self, sine_samples, number_of_actions, lr = None):  # lr=0.00001
        if lr is None:
            lr = self.lr
        if len(glob('model_checkpoint.h5')) and 0:
            model = load_model('model_checkpoint.h5')
        else:
            if 0:
                model = Sequential()
                model.add(Dense(sine_samples, input_dim=sine_samples, activation='tanh'))
                # model.add(Dense(sine_samples, activation='tanh'))
                model.add(Dense(sine_samples // 2, activation='tanh'))
                model.add(Dense(sine_samples // 2, activation='tanh'))
                model.add(Dense(sine_samples // 4, activation='tanh'))
                model.add(Dense(sine_samples // 8, activation='tanh'))
                model.add(Dense(number_of_actions, activation=None))
                model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
            else:
                model = Sequential()
                model.add(Dense(sine_samples//2, input_dim=sine_samples))
                model.add(PReLU())
                # model.add(Dense(sine_samples//4, input_dim=sine_samples))
                # model.add(PReLU())
                model.add(Dense(sine_samples//8, input_dim=sine_samples))
                model.add(PReLU())
                model.add(Dense(number_of_actions, activation=None))
                model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        return model

    def learn_from_memory(self, memory):
        split_row = int(memory.shape[0] * 0.8)
        checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss',
                                                     verbose=0, save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.6,
                                                      patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        hist = CSVLogger('history_adc_output_estimator.csv', separator=',', append=True)
        self.nn.fit(memory.head(split_row).sine.values, memory.head(split_row).actions_rewards.values,
                    validation_data=(memory.head(-split_row).sine.values, memory.head(-split_row).actions_rewards.values),
                    # callbacks=[TQDMNotebookCallback(), reduce_lr, checkpoint, hist],
                    callbacks=[hist, checkpoint, reduce_lr],
                    # callbacks=[hist, checkpoint],
                    epochs=1, batch_size=100, verbose=1)


fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(15, 8))
plt.ion()
plt.show()

dont_train=False
if dont_train and len(glob('model_checkpoint.h5')):
    print('model exist, loading it')
    nn=model(sine_samples=99, number_of_actions=6)
else:
    print('building and loading model')
    e=env()
    nn=model(sine_samples=e.number_of_sars*e.number_of_samples_per_sar, number_of_actions=e.impairments.shape[0]*2)
    mem=q(400)
    rounds=40001
    gamma = 0.2
    codes_max_value_to_plot = 50

    for i in range(-1,rounds) if 0 else tqdm(range(-1,rounds)):
        '''play one move and remember'''
        nn_actions_rewards = nn.nn.predict(e.current_sine.reshape((1, -1)))[0]
        actions_rewards = nn_actions_rewards
        epsilon = (np.random.uniform(0, 1) > 0.2)
        if epsilon:
            chosen_action = np.argmax(actions_rewards)
        else:
            chosen_action = e.random_action_index()
        next_sine, reward = e.get_action_and_update_current_reward_and_next_step(chosen_action)
        next_reward = np.max(nn.nn.predict(next_sine.reshape((1, -1)))[0])
        # if abs(actions_rewards[chosen_action])/np.median(np.abs(actions_rewards))>10:
        #     print('\nreward before env {before}\n after {after}'.format(before = actions_rewards[chosen_action], after = reward + gamma*next_reward))
        nn_estimated_reward=actions_rewards[chosen_action]
        actions_rewards[chosen_action] = reward + gamma*next_reward
        mem.remember(**dict(sine=e.current_sine, actions_rewards=actions_rewards))
        if i % 1 == 0:
            if 0:
                print()
                print(e.impairments.to_frame().T)
                print(e.impairments_in_codes.to_frame().T)
                print('rmse[mv] = {rmse}\nreward = {reward}'.format(reward=e.current_reward, rmse=1e3*np.sqrt(e.mse)))
            action_name = str(e.all_actions.to_frame().iloc[chosen_action].name)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax1.axis([0, 0, -codes_max_value_to_plot, codes_max_value_to_plot])
            e.impairments_in_codes.to_frame().T.stack().reset_index(0, drop=True).T.apply(lambda val: np.clip(val, -codes_max_value_to_plot, codes_max_value_to_plot)). \
                plot.bar(ax=ax1, rot=0, title='rmse[mv] = {rmse: <+10.3g}\nenv current reward = {reward: <+10.3g}\nnn estimated reward = {nn_reward: <+10.3g}\naction = {action}'.
                         format(reward=e.current_reward, nn_reward=nn_estimated_reward,rmse=1e3 * np.sqrt(e.mse), action = action_name))

            ax3.text(x=0.1, y=0.5, s='nn values:\n'+str(e.get_all_actions_rewards_as_table(nn_actions_rewards).to_frame().T.round(4).stack()), family='monospace')
            ax1.grid(True)
            ax2.plot(e.current_sine)
            plt.draw()  # or plt.show(block=False)
            plt.pause(0.0001)

        if i % 20 == 0:
            nn.learn_from_memory(mem.get_data(200))
        if i % 100 == 0:
            '''adapt nn to memory and update memory reward by recursion'''
            print('{:*^100}'.format('resetting sine'))
            print()
            print(pd.concat([e.impairments_in_codes.to_frame('codes').T, e.impairments.to_frame('values').T]))
            print('rmse[mv] = {rmse}\nreward = {reward}'.format(reward=e.current_reward, rmse=1e3 * np.sqrt(e.mse)))
            e.reset_env()


print('starting playing')
random_offset_for_random_state=np.random.normal(0, 0.9, 3)
play=env()
nn_reward=[]
for _ in tqdm(range(2)):
    action=nn.nn.predict(play.current_sine.reshape((1, -1)))
    nn_reward += [np.max(action)]
    play.get_action_and_update_current_reward_and_next_step(np.argmax(action))
pd.DataFrame(nn_reward).plot(title='what nn though the reward was')
# play.plot_history()
