import numpy as np
import copy
import random
import CardList
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch


class ojamamono():
    def __init__(self, net, preprocess) -> None:
        #numpy shape (↓ : →)
        self.board = np.zeros((7, 11), dtype=int) #カードの位置とTypeを記録
        self.boardDirection = np.array([-1 for i in range(7*11)]).reshape((7, 11)) #カードの向きを記録 (0:順転 1:反転)
        self.boardRoad = np.zeros((3*7, 3*11), dtype=int) #実際の通路形状を記録(1カードあたり3*3)
        
        #スタート地点 (Type 20)
        self.SetCard(3, 1, 20, 0) 
        
        #ゴール
        self.goal = [21, 22, 23] #石1, 石2, 金塊
        self.goalDrection = [0, 0, 0, 1, 1, 1] #ゴールの向き決定
        random.shuffle(self.goal) #混ぜる
        random.shuffle(self.goalDrection) #混ぜる
        self.goalDrection = self.goalDrection[:3]
        
        self.RoadBuilder(1, 9, 23, 0)
        self.RoadBuilder(3, 9, 23, 0)
        self.RoadBuilder(5, 9, 23, 0)
        
        #resnet50
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #weights = ResNet18_Weights.DEFAULT
        #model = resnet18(weights=weights)
        #model.fc = nn.Linear(model.fc.in_features, 2)
        #model.eval()
        #self.net = model.to(self.device)
        #self.preprocess = weights.transforms()
        self.net = net
        self.preprocess = preprocess

        
        #デッキ
        self.deck = [
                1, 1, 1, 1, 1,      # 十字路
                2, 2, 2, 2, 2,      # T字路横
                3, 3, 3, 3, 3,      # T字路縦
                4, 4, 4, 4,         # 曲がり１
                5, 5, 5, 5, 5,      # 曲がり２
                6, 6, 6,            # 直線横
                7, 7, 7, 7,         # 直線縦
                8,                  # 十字止
                9,                  # T字止横
                10,                 # T字止縦
                11,                 # 曲止め１
                12,                 # 曲止め２
                13,                 # 直線止横
                14,                 # 直線止縦
                15,                 # 袋小路横
                16]                 # 袋小路縦
        random.shuffle(self.deck) #混ぜる
        
        #手札
        #self.hand = [self.deck.pop() for i in range(random.randint(1, 6))] #妨害カードを考慮
        #self.hand = [self.deck.pop() for i in range(6)]
        self.hand = []
        
        #スコア用の変数
        self.whereisopen = [0, 0, 0] #どのゴールが開いたか
        
        pass
    
    def RoadBuilder(self, y, x, CardType, direction=0):
        
        CardConnect = CardList.rconnect[CardType][direction]
        #初期化
        self.boardRoad[y*3:y*3+3, x*3:x*3+3] = 0
        
        if CardConnect[0] != 0:
            self.boardRoad[y*3+1][x*3] = 1  #左
        if CardConnect[1] != 0:
            self.boardRoad[y*3][x*3+1] = 1  #上
        if CardConnect[2] != 0:
            self.boardRoad[y*3+1][x*3+2] = 1  #右
        if CardConnect[3] != 0:
            self.boardRoad[y*3+2][x*3+1] = 1  #下
        
        if 2 not in CardConnect:
            self.boardRoad[y*3+1][x*3+1] = 1  #真ん中
        #cls.board_Road[y*3+1][x*3+1] = 
    
    def SetCard(self, y, x, CardType, direction=0):
        #カードを置く
        self.RoadBuilder(y, x, CardType, direction)
        self.board[y][x] = CardType
        self.boardDirection[y][x] = direction

    def RemoveCard(self, y, x):
        #カードを取り除く
        self.boardRoad[y*3+1][x*3] = 0
        self.boardRoad[y*3][x*3+1] = 0
        self.boardRoad[y*3+1][x*3+2] = 0
        self.boardRoad[y*3+2][x*3+1] = 0
        self.boardRoad[y*3+1][x*3+1] = 0
        self.board[y][x] = 0
        self.boardDirection[y][x] = -1
    
    def AddSentinel(self, sentinel=-1):
        #番兵を追加したboardRoadのコピーを返す
        board = copy.deepcopy(self.boardRoad)
        sh = np.array([sentinel for i in range(3*7)]).reshape((3*7,1))
        sv = np.array([sentinel for i in range(3*11+2)])
        board = np.hstack([board, sh])
        board = np.hstack([sh, board])
        board = np.vstack([board, sv])
        board = np.vstack([sv, board])
        
        return board
    
    def GetAroundCardExist(self, y, x):
        cardexist = [0, 0, 0, 0] #左、上、右、下
        #どの方角にカードがあるか
        #上下方向の検知
        if y == 0:
            cardexist[1] = -1   #上は壁
            if self.board[y+1][x] != 0: #下のみ検知
                cardexist[3] = 1
        elif y == 6:
            cardexist[3] = -1   #下は壁
            if self.board[y-1][x] != 0: #上のみ検知
                cardexist[1] = 1
        else:
            if self.board[y+1][x] != 0:
                cardexist[3] = 1
            if self.board[y-1][x] != 0:
                cardexist[1] = 1
        
        #左右方向の検知
        if x == 0:
            cardexist[0] = -1   #左は壁
            if self.board[y][x+1] != 0: #右のみ検知
                cardexist[2] = 1
        elif x == 10:
            cardexist[2] = -1   #右は壁
            if self.board[y][x-1] != 0: #左のみ検知
                cardexist[0] = 1
        else:
            if self.board[y][x+1] != 0:
                cardexist[2] = 1
            if self.board[y][x-1] != 0:
                cardexist[0] = 1
        return cardexist
    
    def CheckLocationAvailable(self, cardType, y, x, direct):
        #
        #self.boardを参照 self.boardRoadは参照しない
        #y, xに設置可能ならTrue
        #
        
        #ゴール地点は設置不可能とする
        if x == 9:
            if y == 1 or y == 3 or y == 5:
                return False
        
        #その位置が設置可能か調べる
        connect = CardList.rconnect[cardType][direct]
        cardexist = self.GetAroundCardExist(y, x) #左、上、右、下
        #どの方角にカードがあるか

        #print(f"B {cardexist[1]} B \n{cardexist[0]} P {cardexist[2]} \nB {cardexist[3]} B")
        
        #ほかのカードに隣接していなければFalse
        if 1 not in cardexist:
            return False
        
        #接続可能かチェック
        inverseIndex = [2, 3, 0, 1]
        needOneConnection = False   #最低限1つの接続は必要
        for i, v in enumerate(cardexist):
            if v == 1:
                if i == 0:
                    adjoinCard = CardList.rconnect[self.board[y][x-1]][self.boardDirection[y][x-1]] #隣接カードの情報
                if i == 1:
                    adjoinCard = CardList.rconnect[self.board[y-1][x]][self.boardDirection[y-1][x]] #隣接カードの情報
                if i == 2:
                    adjoinCard = CardList.rconnect[self.board[y][x+1]][self.boardDirection[y][x+1]] #隣接カードの情報
                if i == 3:
                    adjoinCard = CardList.rconnect[self.board[y+1][x]][self.boardDirection[y+1][x]] #隣接カードの情報
                    
                if connect[i] != 0: #もし、i方向が通行可能なら
                    if adjoinCard[inverseIndex[i]] == 0:   #隣接したカードも通行可能でないといけない
                        return False
                    else:
                        needOneConnection  = True
                if connect[i] == 0:#もし、i方向が通行不可なら
                    if adjoinCard[inverseIndex[i]] != 0:   #隣接したカードも通行不可でないといけない
                        return False
        if needOneConnection:
            self.SetCard(y, x, cardType, direct)
            b = self.BFS()
            self.RemoveCard(y, x)
            if b[y*3+1][x*3] != 0 or b[y*3][x*3+1] != 0 or b[y*3+1][x*3+2] != 0 or b[y*3+2][x*3+1] != 0:
                return True
        return False
        
    def GetCandidateList(self, CurrentHand=[]):
        #return 二次元リスト [[y, x, Cardtype, direction]]
        #
        
        Candidate = []
        
        if CurrentHand == []:
            CurrentHand = self.hand
        
        for cType in CurrentHand:
            for direct in range(2):
                for y in range(7):
                    for x in range(11):
                        if self.board[y][x] == 0 and self.CheckLocationAvailable(cType, y, x, direct):
                            Candidate.append([y, x, cType, direct])
        return Candidate
    
    def CardDraw(self):
        if len(self.deck) != 0:
            self.hand.append(self.deck.pop())
    
    def CardThrow(self, cardType):
        self.hand.remove(cardType)
        
    def BFS(self):
        
        #BFSによる探索を行う
        y_s =  3*3+1+1  #スタート位置
        x_s =  1*3+1+1
        que = [[y_s, x_s]]
        checkBoard = self.AddSentinel() #番兵を追加した確認用の盤面
        checkBoard[y_s][x_s] = 2  #1を通路として使ってる関係上 2を用いる
        
        #保険の処理 (ゴール地点の初期化)
        #pass
        #
        
        while len(que) != 0:
            look = que.pop(0)   #現在の探索地点
            y = look[0]
            x = look[1]
            dis = int(checkBoard[y][x])
            
            #未探索の通行可能位置
            mov = []
            if checkBoard[y][x-1] == 1: #左
                mov.append([y, x-1])
            if checkBoard[y-1][x] == 1: #上
                mov.append([y-1, x])
            if checkBoard[y][x+1] == 1: #右
                mov.append([y, x+1])
            if checkBoard[y+1][x] == 1: #下
                mov.append([y+1, x])
            
            for m in mov:
                if checkBoard[m[0]][m[1]] == 1:
                    checkBoard[m[0]][m[1]] = str(dis+1)
                    que.append(m)
        
        #ボードの成形を行う
        #スタートのコストを2としたため、ここで1を引く。残っていた通路もここの処理で消える
        checkBoard = np.where(checkBoard > 0, checkBoard - 1, checkBoard)   
        #番兵の削除
        checkBoard = checkBoard[1:-1,1:-1]
        return checkBoard
    
    def openGoalCard(self):
        for y, x, n in zip([1, 3, 5],[9, 9, 9], range(3)):
            for j, i in zip([0, -1, 0, 1],[-1, 0, 1, 0]):
                if self.board[y+j][x+i] != 0 and self.whereisopen[n] == 0:
                    self.SetCard(y, x, self.goal[n], self.goalDrection[n])
                    self.whereisopen[n] = 1
                    break
        pass
    
    def GoalCheck(self):
        self.openGoalCard()
        BFSboard = self.BFS()
        if BFSboard[1*3+1][9*3+1] and self.board[1][9] == 23:
            return 1
        if BFSboard[3*3+1][9*3+1] != 0 and self.board[3][9] == 23:
            return 2
        if BFSboard[5*3+1][9*3+1] != 0 and self.board[5][9] == 23:
            return 3
        
        return 0
    
    def GetManhattan(self, y_br, x_br, goalInfo=[]):
        dis = [10e6, 10e6, 10e6]
        if goalInfo==[]:
            if self.whereisopen[0] == 0 or (self.whereisopen[0] == 1 and self.goal[0] == 23):
                dis[0] = abs(3*1+1 - y_br) + abs(3*9+1 - x_br)
            if self.whereisopen[1] == 0 or (self.whereisopen[1] == 1 and self.goal[1] == 23):
                dis[1] = abs(3*3+1 - y_br) + abs(3*9+1 - x_br)
            if self.whereisopen[2] == 0 or (self.whereisopen[2] == 1 and self.goal[2] == 23):
                dis[2] = abs(3*5+1 - y_br) + abs(3*9+1 - x_br)
        else:
            if goalInfo[0] == 1:
                dis[0] = abs(3*1+1 - y_br) + abs(3*9+1 - x_br)
            elif goalInfo[1] == 1:
                dis[1] = abs(3*3+1 - y_br) + abs(3*9+1 - x_br)
            elif goalInfo[2] == 1:
                dis[2] = abs(3*5+1 - y_br) + abs(3*9+1 - x_br)
            else:
                if goalInfo[0] == 0:
                    dis[0] = abs(3*1+1 - y_br) + abs(3*9+1 - x_br)
                if goalInfo[1] == 0:
                    dis[1] = abs(3*3+1 - y_br) + abs(3*9+1 - x_br)
                if goalInfo[2] == 0:
                    dis[2] = abs(3*5+1 - y_br) + abs(3*9+1 - x_br)
        return dis
    
    
    def GetBoardScore(self, candidate, goalInfo = [], none_nn=False):
        # candidate [y, x, Cardtype, direction]
        #0: ゴールまでの最短マンハッタン距離
        #1: 現ボードにおけるスタートにつながった接続可能数
        #2: 接続可能地点のx座標総計
        #-: 設置するカードの分岐数 -
        #3: 通過可能な道の総数
        #-: 通行不可能な道の総数 -
        #4: 設置するカードのx座標
        #-: 上下のリミットまでの距離 -
        #5: その行に接続可能点はあるか? 0 - 1
        #6: resnet50
        #7: resnet50
        
        if candidate != []:
            self.SetCard(candidate[0], candidate[1], candidate[2], candidate[3])
        evalBoard = self.BFS()
        
        cardType = candidate[2]
        direction = candidate[3]
        
        #0 : 最短マンハッタン距離
        minManhattan = 10e6
        mTemp = 0
        for y in range(7):
            for x in range(11):
                if evalBoard[3*y+1][3*x+1] != 0:    #通行可能カードか？
                    cardexit = self.GetAroundCardExist(y, x)
                    #条件 : その方向にカードが存在しない and その方向への通路を持っている
                    if cardexit[0] == 0 and evalBoard[3*y+1][3*x] != 0 and x != 0:   #左 x!=0は一番左からは接続できないため
                        mTemp = min(self.GetManhattan(3*y+1, 3*x - 2, goalInfo))                            #-2は設置予定個所の中心からの距離を測るため
                        minManhattan = min(mTemp, minManhattan)
                        pass
                    if cardexit[1] == 0 and evalBoard[3*y][3*x+1] != 0 and y != 0:   #上
                        mTemp = min(self.GetManhattan(3*y - 2, 3*x+1, goalInfo))
                        minManhattan = min(mTemp, minManhattan)
                        pass
                    if cardexit[2] == 0 and evalBoard[3*y+1][3*x+2] != 0 and x != 10: #右
                        mTemp = min(self.GetManhattan(3*y+1, 3*x+2 + 2, goalInfo))
                        minManhattan = min(mTemp, minManhattan)
                        pass
                    if cardexit[3] == 0 and evalBoard[3*y+2][3*x+1] != 0 and y != 6: #下
                        mTemp = min(self.GetManhattan(3*y+2 + 2, 3*x+1, goalInfo))
                        minManhattan = min(mTemp, minManhattan)
                        pass
                    pass
        
        #1: 現ボードにおけるスタートにつながった接続可能数
        connectionNum = 0
        connection_count_map = np.zeros((7, 11), dtype=int)
        for y in range(7):
            for x in range(11):
                if evalBoard[3*y+1][3*x+1] != 0:    #通行可能カードか？
                    cardexit = self.GetAroundCardExist(y, x)
                    cardInfo = CardList.rconnect[self.board[y][x]][self.boardDirection[y][x]]
                    for i, j, k in zip(range(4), [0, -1, 0, 1], [-1, 0, 1, 0]):
                        if cardexit[i] == 0 and cardInfo[i] == 1:
                            connection_count_map[y+j][x+k] = 1
        connectionNum = np.count_nonzero(connection_count_map)
        
        #2: 接続可能地点のx座標総計
        connectablePosSum = 0
        for y in range(7):
            for x in range(11):
                if connection_count_map[y][x] == 1:
                    connectablePosSum += x
        
        #-: 設置するカードの分岐数
        #branchNum = 0
        #if 2 in CardList.rconnect[cardType][direction]:
        #    branchNum = 0
        #else:
        #    branchNum = np.count_nonzero(CardList.rconnect[cardType][direction]) - 1 
            
        #3: 通過可能な道の総数
        NumOfPass = np.count_nonzero(evalBoard)
        
        #-: 通過不可能な道の総数
        #NumOfNonePass = np.count_nonzero(self.boardRoad) - NumOfPass
        
        #4: 設置するカードのx座標
        setposx = candidate[1]
        
        #-
        #ydis = min(candidate[0], 6-candidate[0])
        
        #5：その行に接続可能点はあるか? 0:1
        row_connect = 0
        for x0 in range(candidate[1]-1,11):
            if x0 < 0:
                pass
            elif connection_count_map[candidate[0],x0] == 1:
                row_connect = 1
                break
        
        #net 6, 7
        if none_nn == False:
            input_board_dim1 = copy.deepcopy(self.boardRoad)
            input_board_dim1 *= 255
            input_board_dim2 = np.zeros((3*7, 3*11))
            input_board_dim3 = np.zeros((3*7, 3*11))
            for y1 in range(7):
                for x1 in range(11):
                    if connection_count_map[y1,x1] == 1:
                        input_board_dim2[y1*3:y1*3+3, x1*3:x1*3+3] = 255
            
            for y2 in range(7):
                for x2 in range(11):
                    if self.board[y2, x2] > 7 and self.board[y2, x2] < 17:
                        input_board_dim3[y2*3:y2*3+3, :] = 255
            
            img = torch.tensor(np.array([input_board_dim1, input_board_dim2, input_board_dim3]), dtype=torch.int8).to(self.device)
            img_transformed = self.preprocess(img.unsqueeze(0))
            resout = self.net(img_transformed)
            resout = [resout[0][0].item(), resout[0][1].item()]
        else:
            resout = [0, 0]
        
        if candidate != []:
            self.RemoveCard(candidate[0], candidate[1])

        #return [minManhattan, connectionNum, connectablePosSum, branchNum, NumOfPass, NumOfNonePass, setposx, ydis]
        return [minManhattan, connectionNum, connectablePosSum, NumOfPass, setposx, row_connect, resout[0], resout[1]]

    @staticmethod
    def DebugShowCardShape(cardType, direct=0):
        c = CardList.rconnect[cardType][direct]
        print("------")
        print(f"0 {c[1]} 0 \n{c[0]} 0 {c[2]} \n0 {c[3]} 0")
        print("------")
    
    def DebugRandomPlay(self, turn = 9999):
        T = 0
        self.__init__()
        while T < turn:
            candidate = self.GetCandidateList()
            random.shuffle(candidate)
            
            if candidate == [] and len(self.deck) == 0:
                #####################
                #print(f"LOSE")
                return 0
                break
            elif candidate == []:
                self.CardDraw()
            else:
                print(f"c: {candidate[0]}  s : {self.GetBoardScore(candidate[0])} t:{T}")
                y = candidate[0][0]
                x = candidate[0][1]
                c = candidate[0][2]
                d = candidate[0][3]
                
                self.CardThrow(c)
                self.SetCard(y, x, c, d)
                self.CardDraw()
            
            if self.GoalCheck() != 0:
                ##############################
                #print("WIN")
                return 1
                break
            T += 1
                

if __name__ == "__main__":
    t = ojamamono()
    #t.SetCard(3, 2, 2, 1) #y:3, x:2にCardType=2(T字路横)を向き1(反転)で設置
    t.SetCard(3, 2, 6, 0) 
    t.SetCard(3, 3, 6, 0) 
    #t.SetCard(3, 4, 6, 0) 
    #t.SetCard(3, 5, 6, 0)
    t.SetCard(3, 6, 6, 0) 
    t.SetCard(4, 1, 8, 0) 
    #t.GoalCheck()
    
    ct = 1
    
    win = 0
    #for i in range(1000):
    #    if i % 100 == 0:
    #        print(i)
    #    win += t.DebugRandomPlay(1000)
    #t.DebugRandomPlay(1000)
    #t.DebugRandomPlay(1000)
    #print(f"win : {win}  {win/10}%")
    #t.GoalCheck()
    #c = t.GetCandidateList([1])
    #random.shuffle(c)
    #print(c[0])
    print(t.GetBoardScore([3, 7, 1, 0]))

    
    for y in range(7):
        for x in range(11):
            #print(f"y:{y} x:{x}={t.CheckLocationAvailable(ct, y,x, 0)}")
            if t.board[y][x] != 0:
                print("C ", end="")
                pass
            elif t.CheckLocationAvailable(ct, y,x, 0):
                print("■ ", end="")
            else:
                print("□ ", end="")
        print("")
    
    bfs = t.BFS()

    with open('test.txt', 'w', encoding='utf-8') as f:
        for d in bfs:
            for c in d:
                f.write(f"{c:_=2} ")
            f.write("\n")
    with open('test1.txt', 'w', encoding='utf-8') as f:
        for d in t.boardRoad:
            for c in d:
                if c == 0:
                    f.write("□ ")
                elif c == 99:
                    f.write("● ")
                else:
                    f.write("■ ")
            f.write("\n")
    pass