import numpy as np
from queue import Queue
c1=-0.510066
c2=0.760666
c3=-0.35663
c4=-0.184483
class HeuristicModel():

    def __init__(self, board,piece,origin):
        self.board = board
        self.piece = piece
        self.origin = origin

    def generateMoveQueue(self, bestCol,maxRot,pieceLeftCol):
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

    def determineOptimalMove(self):
        board = self.board
        piece = self.piece
        origin = self.origin
        maxScore = -999
        bestCol = 0
        maxRot = 0
        maxLeftCol=0
        nonRegularizedPiece=np.copy(piece)
        nonRegularizedOrigin=np.copy(origin)
        for i in range(4):
            piece,origin,pieceLeftCol=self.regularizePiece(np.copy(nonRegularizedPiece),np.copy(nonRegularizedOrigin))
            score,loc = self.determineOptimalColumn(board,piece)
            nonRegularizedPiece =self.rotate(nonRegularizedPiece,nonRegularizedOrigin)
            if score > maxScore:
                maxScore = score
                bestCol=loc
                maxRot = i
                maxLeftCol=pieceLeftCol
        return self.generateMoveQueue(bestCol,maxRot,maxLeftCol)

    #checked
    def rotate(self,piece,origin):
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
        return piece

    def regularizePiece(self,piece,origin):
        xDiff=999
        yDiff=999
        for i in range(piece.shape[0]):
            xDiff=min(xDiff,piece[i,0])
            yDiff=min(yDiff,piece[i,1])
        origin[0]-=xDiff
        origin[1]-=yDiff
        for i in range(piece.shape[0]):
            piece[i,0]-=xDiff
            piece[i,1]-=yDiff

        return piece.astype(int),origin,yDiff.astype(int)

    def determineOptimalColumn(self, board,piece):
        maxScore = -999.0
        maxLoc = 0

        for column in range(10):
            score = self.determineScoreInColumn(board,piece,column)

            if score > maxScore:
                maxScore = score
                maxLoc = column
        return maxScore,maxLoc


    def determineScoreInColumn(self, board,piece,column):
        row = self.determineRowWithinColumn(board,piece,column)
        #print("row",row)
        return self.calculateScore(board,piece,row,column)

    def determineRowWithinColumn(self, board,piece,column):
        for row in range(20):
            for j in range(piece.shape[0]):

                if 19-row+piece[j,0]>19:
                    break
                elif column+piece[j,1]>9 :
                    return 0
                elif board[19-row+piece[j,0],column+piece[j,1]]==1:
                    return (19-row)+1
        return 0



    def calculateScore(self, board,piece,row,column):
        tempBoard,valid=self.addToBoard(np.copy(board),piece,row,column)
        score = -999
        if valid:
            score=0
            lines =self.linesCleared(tempBoard)
            height = (self.aggHeight(tempBoard)-lines)
            holes = self.numHoles(tempBoard)
            bump=self.bumpiness(tempBoard)

            score += height*c1+holes*c3+lines*c2+bump*c4
            #print(row,column,lines,height,holes,bump,"%.2f" % score)

        return score

    def addToBoard(self, board,piece,row,column):
        for i in range(piece.shape[0]):


            if piece[i,0]+row<20 and piece[i,1]+column<10 and (not board[piece[i,0]+row,piece[i,1]+column]==1):
                board[piece[i,0]+row,piece[i,1]+column]=1

            else:
                return board,False
        #print(board)
        return board,True

    #checked
    def aggHeight(self, board):
        topRow = self.getTopRow(board)

        maxHeight =0
        for i in range(10):
            if topRow[i]!=-1:
                maxHeight+=topRow[i]
        maxHeight+=1
        return maxHeight

    #checked sorta
    def numHoles(self, board):
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
    def linesCleared(self, board):
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

    def bumpiness(self, board):
        boardRow = self.getTopRow(board)
        totalSum=0
        for i in range(10-1):
            totalSum+=abs(boardRow[i]-boardRow[i+1])
        return totalSum

    #checked
    def getTopRow(self, board):
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
