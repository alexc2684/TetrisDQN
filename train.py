import gym
import math
import random
import numpy as np
import matplotlib
import socket
import time
import sys
from threading import Thread
from time import sleep
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from DQN import DQN
from ReplayMemory import ReplayMemory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 50
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(7)]])

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

last_sync = 0

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

featureString = ""

def serverPython():

    while(True):
        featureString = client_socket.recv(2048).decode('utf-8')
        if(featureString==''):
            sys.exit()

def clientPython():

    while(True):

        #test = raw_input('send here ')
        if(test=="q"):
            sys.exit()
        test=test + '\n'
        bytesTest=test.encode('utf-8')
        #client_socket.sendall(bytesTest)
        #print ('you sent : ' , test)


def initSocket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("", 5000))
    thread = Thread(target = serverPython, args = ())
    thread2 = Thread(target = clientPython, args = ())
    thread.start()
    thread2.start()
    thread2.join()
    thread.join()

def parseStream(stream):
    stream = stream.split(',')
    stream = list(map(int, stream))
    done = Tensor(stream[0])
    reward = Tensor(stream[1])
    state = torch.Tensor(stream[2:len(stream)-1])
    time = stream[len(stream)-1]
    return done, state, reward, time

def train(num_episodes):
    initSocket()
    while featureString != '':
        time.sleep(1)
    for i in range(num_episodes):
        print(featureString)
        done, state, reward, time = parseStream(featureString)
        #wait for game to start
        #TODO: have socket send when it is connected
        time = 0
        last = torch.zeros((214, 1))
        curr = torch.zeros((214, 1))
        curr_state = curr - last
        while not done:
            action = select_action(curr_state)
            #TODO: map outputs to strings, to socket
            print()
            done, curr, reward, time = parseStream(featureString)
            last = curr
            if not done:
                next_state = curr - last
            else:
                next_state = None

            memory.push(curr_state, action, next_state, reward)

            curr_state = next_state
            optimize_model()
            if done:
                episode_durations.append(reward)
                plot_durations()
                break

    print("OPTIMIZED")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train(1)




    #read socket
    #wait for signal to init
    #while init signal is True
    #feed byte stream into helper method
    # state, reward, done = #helper converter to (201, 1) Tensor
    #observe new state as difference between curr and old
    #push to replay memory
    #OPTIMIZE
