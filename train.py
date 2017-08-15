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
from queue import Queue

import torch
from torch import Tensor, LongTensor, FloatTensor, ByteTensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from DQN import DQN

from ReplayMemory import ReplayMemory
from HeuristicModel import HeuristicModel

Tensor = FloatTensor
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 150
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05

MAX_STEPS= 10000
steps_done = 0
DECAY_RATE = 4
newPiece=True
useHeuristic=True
newModel=True
model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)
moveQueue= Queue()

def getEpsilonThreshold():
    global steps_done,EPS_END,EPS_START,DECAY_RATE,MAX_STEPS
    return max(EPS_END,EPS_END + (EPS_START - EPS_END)*(1-steps_done/(DECAY_RATE*MAX_STEPS)))

def select_action(state,board, piece,origin):
    global steps_done,moveQueue,newPiece,useHeuristic
    steps_done += 1

    if not moveQueue.isEmpty():
        return LongTensor([[moveQueue.dequeue()]])

    sample = random.random()
    eps_threshold = getEpsilonThreshold()

    if useHeuristic and newPiece and sample<eps_threshold:
        heuristic = HeuristicModel(board,piece,origin)
        moveQueue= heuristic.determineOptimalMove()
        return LongTensor([[moveQueue.dequeue()]])
    newPiece = False
    sample = random.random()
    if sample > eps_threshold:
        print("DQN Decision - Epsilon value: ", eps_threshold)
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
    temp=model(state_batch)
    state_action_values = temp.gather(1, action_batch.view(-1,1))

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))

    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

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
    stream = stream[:217]
    length = len(stream)
    stream = list(map(float, stream))
    origin=np.array(stream[length-2:length])

    stream = stream[:215]
    length = len(stream)
    #print(stream)



    stream = list(map(int, stream))

    done = stream[0]
    reward = LongTensor([stream[1]])
    board = np.array(stream[2:202])
    board = board.reshape((20, 10)) #TODO: make sure it formats correctly
    # board = np.flip(board, 0)
    currPiece = np.array(stream[length-8:length])
    i = 0
    pieceArr = np.zeros((4,2))
    while i < len(currPiece):
        x = currPiece[i]
        y = currPiece[i+1]
        pieceArr[(int) (i/2),0] = x
        pieceArr[(int) (i/2),1] = y
        if x < 20 and y < 10:
            board[x, y] = 2

        i += 2
    pieceArr.astype(int)


    state = np.expand_dims(board, axis=0)
    state = torch.from_numpy(board)

    while i < len(pieceArr):

        if pieceArr[i,0] < 20 and pieceArr[i,1] < 10:
            board[pieceArr[i,0], pieceArr[i,1]] = 0

    state = state.unsqueeze(0).type(Tensor)

    return done, state, board, reward, pieceArr, origin, True

    print("Error: Bad socket read")
    return False, 0,0,0, 0,0, False

def translateAction(n):
    global newPiece
    if n == 0:
        return "left"
    elif n == 1:
        return "right"
    elif n == 2:
        return "up"
    elif n == 3:
        newPiece=True
        return "space"
    elif n == 4:
        return "z"
    elif n == 5:
        return "down"
    elif n == 6:
        return "shift"



def sendText(text):
    text=text + '\n'
    bytesTest=text.encode('utf-8')
    client_socket.sendall(bytesTest)

def receiveNextFeatureString():
    featureString = ""
    featureString = client_socket.recv(2048).decode('utf-8')
    while not featureString: #or len(featureString)<400:
        sendText("empty")
        featureString = client_socket.recv(2048).decode('utf-8')
    return featureString

def train(num_episodes):
    global newPiece, model,newModel
    client_socket.connect(("", 5000))
    #featureString = client_socket.recv(2048).decode('utf-8')
    featureString = ""
    if not newModel:
        model = torch.load('model.pkl')
    counter = 0
    for i in range(num_episodes):
        # #
        print("Episode: " + str(i+1) + " ", end = "\t")
        print(u'ε',end=" ")
        eps_threshold=getEpsilonThreshold()
        print(round(eps_threshold*1000)/1000,end = "  \t")
        if i % 10 ==0 and i>=0:
            torch.save(model.state_dict(),'model.pkl')
        featureString=receiveNextFeatureString()
        done, state, board, reward, piece, origin, didReceive = parseStream(featureString)

        last = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr_state =  curr - last
        curr_state = curr_state.long()
        newPiece = True
        while not done:
            counter +=1

            action = select_action(curr_state,board,piece,origin)
            actionText = translateAction(action[0][0])
            sendText(actionText)
            #TODO: map outputs to strings, to socket
            last = curr
            featureString = receiveNextFeatureString()
            done, state, board, reward, piece, origin, didReceive = parseStream(featureString)

            if didReceive:
                penalty = 0
                gameOverPenalty = 200
                if not done:
                    next_state = curr - last
                    next_state = next_state.float()
                else:
                    next_state = None
                    penalty+=gameOverPenalty

                curr_state=curr_state.long()
                if action[0][0] != 3 :
                    penalty += 2
                penalty = LongTensor([penalty])
                reward -= penalty
                action = action.long()

                memory.push(curr_state.float(), action, next_state, reward.float())


                curr_state = next_state
                optimize_model()
            else:
                sendText("empty")
            if done:
                episode_durations.append(reward)
                sendText("reset")
                print("Score: " + str(reward[0]+gameOverPenalty))
                break



    #print("OPTIMIZED")
    plt.ioff()
    plt.show()
    sendText("end")



if __name__ == "__main__":
    train(200)







    #read socket
    #wait for signal to init
    #while init signal is True
    #feed byte stream into helper method
    # state, reward, done = #helper converter to (201, 1) Tensor
    #observe new state as difference between curr and old
    #push to replay memory
    #OPTIMIZE
