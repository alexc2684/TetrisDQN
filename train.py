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
from torch import Tensor, LongTensor, FloatTensor, ByteTensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from DQN import DQN
from ReplayMemory import ReplayMemory

Tensor = FloatTensor
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 150
GAMMA = 0.9990
EPS_START = 0.95
EPS_END = 0.05
#EPS_DECAY = 100
MAX_STEPS= 10000
steps_done = 0
DECAY_RATE = 3
model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

def select_action(state):
    global steps_done
    sample = random.random()

    eps_threshold = max(EPS_END,EPS_END + (EPS_START - EPS_END)*(1-steps_done/(DECAY_RATE*MAX_STEPS)))
    # print(eps_threshold)
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(6)]])

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
    temp=model(state_batch)
    state_action_values = temp.gather(1, action_batch.view(-1,1))

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))

    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    #print(loss)
    # print("Loss: " + str(((loss.data)[0])/steps_done))
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



def parseStream(stream):
    wholeStream = stream
    stream = stream[:stream.find('\\')]
    stream = stream.split(',')
    stream = stream[:215]
    length = len(stream)
    #print(stream)
    #print(len(stream))
    try:
        stream = list(map(int, stream))
    except:
        print(stream)
        print(wholeStream)
    done = stream[0]
    #print(stream[1])
    reward = LongTensor([stream[1]])
    board = np.array(stream[2:202])
    board = board.reshape((20, 10)) #TODO: make sure it formats correctly
    # board = np.flip(board, 0)
    currPiece = np.array(stream[length-8:length])
    i = 0
    while i < len(currPiece):
        y = currPiece[i] - 1
        x = currPiece[i+1] - 1
        if x < 10 and y < 20:
            board[y, x] = 1
        i += 2

    # print(board)
    board = np.expand_dims(board, axis=0)
    board = torch.from_numpy(board)
    board = board.unsqueeze(0).type(Tensor)
    state = np.array(stream[2:len(stream)]) #TODO: make matrix/check dims
    # print(state)
    state = np.reshape(state, (1, 213))
    state = torch.from_numpy(state)
    #print("curr state type:")
    #print(type(state))
    return done, board, reward

def translateAction(n):
    if n == 0:
        return "left"
    elif n == 1:
        return "right"
    elif n == 2:
        return "up"
    elif n == 3:
        return "down"
    elif n == 4:
        return "z"
    elif n == 5:
        return "space"
    elif n == 6:
        return "shift"


def sendText(text):
    text=text + '\n'
    bytesTest=text.encode('utf-8')
    client_socket.sendall(bytesTest)

def receiveNextFeatureString():
    featureString=""
    featureString = client_socket.recv(2048).decode('utf-8')
    while not featureString  : #or len(featureString)<400:
        sendText("empty")
        featureString = client_socket.recv(2048).decode('utf-8')
    return featureString

def train(num_episodes):
    client_socket.connect(("", 5000))
    #featureString = client_socket.recv(2048).decode('utf-8')
    featureString=""
    for i in range(num_episodes):
        # #
        print("Episode: " + str(i+1))
        featureString=receiveNextFeatureString()
        done, state, reward = parseStream(featureString)

        last = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr_state =  curr - last
        curr_state = curr_state.long()
        while not done:
            action = select_action(curr_state)[0][0]
            actionText = translateAction(action)
            sendText(actionText)
            #TODO: map outputs to strings, to socket
            last = curr
            featureString=receiveNextFeatureString()
            done, curr, reward = parseStream(featureString)
            if not done:
                next_state = curr - last
                next_state = next_state.float()
            else:
                next_state = None
            curr_state=curr_state.long()

            action=Tensor([action]).long()
            #print(action.size())
            memory.push(curr_state.float(), action, next_state, reward.float())

            curr_state = next_state
            optimize_model()

            if done:
                episode_durations.append(reward)
                sendText("reset")
                print("Score: "+ str(reward[0]))
                break



    #print("OPTIMIZED")
    plt.ioff()
    plt.show()
    sendText("end")

if __name__ == "__main__":
    train(100)




    #read socket
    #wait for signal to init
    #while init signal is True
    #feed byte stream into helper method
    # state, reward, done = #helper converter to (201, 1) Tensor
    #observe new state as difference between curr and old
    #push to replay memory
    #OPTIMIZE
