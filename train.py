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
from queue import Queue
from ReplayMemory import ReplayMemory

Tensor = FloatTensor
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 150
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
#EPS_DECAY = 100
MAX_STEPS= 10000
steps_done = 0
DECAY_RATE = 2
newPiece=True
model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)
moveQueue= Queue()

def select_action(state,board, piece,origin,newEpisode):
    global steps_done,moveQueue,newPiece
    steps_done += 1
    if not moveQueue.isEmpty():
        #time.sleep(0.1)
        return LongTensor([[moveQueue.dequeue()]])
    if steps_done<0:
        s= input("Input move: ")
        action = 0
        if s=="a":
             action=0
        elif s=="d":
             action=1
        elif s=="w":
             action=2
        elif s==" ":
             action=3
        elif s=="z":
             action=4

        return LongTensor([[action]])

    sample=random.random()
    eps_threshold = max(EPS_END,EPS_END + (EPS_START - EPS_END)*(1-steps_done/(DECAY_RATE*MAX_STEPS)))
    if newEpisode:
        print(eps_threshold)
    if newPiece and sample<eps_threshold:

        moveQueue= determineOptimalMove(board,piece,origin)
        time.sleep(0.5)
        #s=input("enter when ready")
        return LongTensor([[moveQueue.dequeue()]])
    newPiece = False
    sample=random.random()
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
    if n == 0:
        return "left"
    elif n == 1:
        return "right"
    elif n == 2:
        return "up"
    elif n == 3:
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
    global newPiece
    client_socket.connect(("", 5000))
    #featureString = client_socket.recv(2048).decode('utf-8')
    featureString = ""
    counter = 0
    for i in range(num_episodes):
        # #
        print("Episode: " + str(i+1) + " ", end = "")
        print(u'ε',end=" ")

        featureString=receiveNextFeatureString()
        done, state, board, reward, piece, origin, didReceive = parseStream(featureString)

        last = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr = torch.zeros((1, 20, 10)).long().unsqueeze(0).type(Tensor)
        curr_state =  curr - last
        curr_state = curr_state.long()
        newEpisode = True
        while not done:
            counter +=1
            newPiece = True
            action = select_action(curr_state,board,piece,origin,newEpisode)
            newEpisode = False
            actionText = translateAction(action[0][0])
            sendText(actionText)
            #TODO: map outputs to strings, to socket
            last = curr
            featureString = receiveNextFeatureString()
            done, state, board, reward, piece, origin, didReceive = parseStream(featureString)

            if didReceive:
                penalty=0
                gameOverPenalty=200
                if not done:
                    next_state = curr - last
                    next_state = next_state.float()
                else:
                    next_state = None
                    penalty+=gameOverPenalty

                curr_state=curr_state.long()
                if action[0][0] !=3 :
                    penalty +=4
                penalty=LongTensor([penalty])
                reward-=penalty
                action=action.long()

                memory.push(curr_state.float(), action, next_state, reward.float())


                curr_state = next_state
                optimize_model()

            if done:
                episode_durations.append(reward)
                sendText("reset")
                print("Score: "+ str(reward[0]+gameOverPenalty))
                break



    #print("OPTIMIZED")
    plt.ioff()
    plt.show()
    sendText("end")


c1=-0.510066
c2=0.760666
c3=-0.35663
c4=-0.184483
def generateMoveQueue(bestCol,maxRot,pieceLeftCol):
    moveQueue = Queue()
    #print("columns!!!!!",bestCol,maxRot,pieceLeftCol)
    if maxRot==3:
        moveQueue.enqueue(4) #z
    else:
        for _ in range(maxRot):
            moveQueue.enqueue(2) #up

    for _ in range(pieceLeftCol-bestCol):
        moveQueue.enqueue(0)

    for _ in range(bestCol-pieceLeftCol):
        moveQueue.enqueue(1)
    moveQueue.enqueue(3)
    return moveQueue

def determineOptimalMove(board, piece,origin):

    maxScore = -999
    bestCol = 0
    maxRot = 0
    maxLeftCol=0
    nonRegularizedPiece=np.copy(piece)
    nonRegularizedOrigin=np.copy(origin)
    for i in range(4):

        piece,origin,pieceLeftCol=regularizePiece(np.copy(nonRegularizedPiece),np.copy(nonRegularizedOrigin))
        #print(piece)
        score,loc = determineOptimalColumn(board,piece)
        #print("ROTATED")
        nonRegularizedPiece =rotate(nonRegularizedPiece,nonRegularizedOrigin)
        #print("score and loc: " ,score,loc)

        if score > maxScore:
            maxScore = score
            bestCol=loc
            maxRot = i
            maxLeftCol=pieceLeftCol
        #print("best col, max rot, leftCol", bestCol,maxRot,pieceLeftCol)
    return generateMoveQueue(bestCol,maxRot,maxLeftCol)

#checked
def rotate(piece,origin):
    #print(origin,origin[0],origin[1])
    for i in range(piece.shape[0]):
        x = piece[i,0];
        y = piece[i,1];
        x -= origin[0]
        y -= origin[1]
        temp = x;
        x = -1 * y;
        y = temp;
        x += origin[0];
        y += origin[1];
        piece[i,0]=x
        piece[i,1]=y


        #print("rotated", x,y)
    return piece

def regularizePiece(piece,origin):
    xDiff=999
    yDiff=999
    for i in range(piece.shape[0]):
        xDiff=min(xDiff,piece[i,0])
        yDiff=min(yDiff,piece[i,1])
    origin[0]-=xDiff
    origin[1]-=yDiff
    #print("piece locs:")
    for i in range(piece.shape[0]):
        piece[i,0]-=xDiff
        piece[i,1]-=yDiff

    return piece.astype(int),origin,yDiff.astype(int)

def determineOptimalColumn(board,piece):
    maxScore = -999.0
    maxLoc = 0

    for column in range(10):
        score = determineScoreInColumn(board,piece,column)

        if score > maxScore:
            maxScore = score
            maxLoc = column
    return maxScore,maxLoc


def determineScoreInColumn(board,piece,column):
    row = determineRowWithinColumn(board,piece,column)
    #print("row",row)
    return calculateScore(board,piece,row,column)

def determineRowWithinColumn(board,piece,column):
    for row in range(20):
        for j in range(piece.shape[0]):

            if 19-row+piece[j,0]>19:
                break
            elif column+piece[j,1]>9 :
                return 0
            elif board[19-row+piece[j,0],column+piece[j,1]]==1:
                return (19-row)+1
    return 0



def calculateScore(board,piece,row,column):
    tempBoard,valid=addToBoard(np.copy(board),piece,row,column)
    score = -999
    if valid:
        score=0
        lines =linesCleared(tempBoard)
        height = (aggHeight(tempBoard)-lines)
        holes = numHoles(tempBoard)
        bump=bumpiness(tempBoard)

        score += height*c1+holes*c3+lines*c2+bump*c4
        #print(row,column,lines,height,holes,bump,"%.2f" % score)

    return score

def addToBoard(board,piece,row,column):
    #print(board)
    for i in range(piece.shape[0]):


        if piece[i,0]+row<20 and piece[i,1]+column<10 and (not board[piece[i,0]+row,piece[i,1]+column]==1):
            board[piece[i,0]+row,piece[i,1]+column]=1

        else:
            return board,False
    #print(board)
    return board,True

#checked
def aggHeight(board):
    topRow = getTopRow(board)

    maxHeight =0
    for i in range(10):
        if topRow[i]!=-1:
            maxHeight+=topRow[i]
    maxHeight+=1
    return maxHeight

#checked sorta
def numHoles(board):
    count = 0
    for col in range(10):
        colContainsBlock=False
        for row in range(20):
            if board[19-row,col]==1:
                colContainsBlock=True
            elif colContainsBlock:
                count +=1
    return count

#pseudo/sudo checked @alex chan
def linesCleared(board):
    lines=0
    for row  in range(20):
        prod = 1
        totalSum = 0
        for col in range(10):
            prod *=board[row,col]
            totalSum+=board[row,col]
        if prod != 0:
            lines +=1

        elif totalSum==0:
            return lines
    return lines

def bumpiness(board):
    boardRow = getTopRow(board)
    totalSum=0
    for i in range(10-1):
        totalSum+=abs(boardRow[i]-boardRow[i+1])
    return totalSum

#checked
def getTopRow(board):
    topRow = []
    for col  in range(10):
        flag = False
        for row in range(20):
            if board[19-row,col]==1:
                topRow.append(19-row)
                flag = True
                break
        if not flag:
            topRow.append(-1)
    return topRow


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
