from GameMain import ojamamono
from network import Network, Network2
import random
import numpy as np
from deap import base, creator, tools
import torch
import multiprocessing
import csv
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch
import copy

# GAパラメータ
GEN_LENGTH = 8*4 + 4 + 4
MU = 0.0
SIGMA = 3
MAX_GEN_VAL = 1.0
MIN_GEN_VAL = -1.0

N_GEN = 500
POP_SIZE = 100
CX_PB = 0.5
MUT_PB = 0.25

#ResNet設定
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.eval()
net_res = model.to(device)
preprocess = weights.transforms()



def reward(object : ojamamono):
    score = 0.0
    score += sum(object.whereisopen)*0.1 #ゴールが開かれると加算
    score += 3 if object.GoalCheck() != 0 else -3   #ゴールすると加算
    bfs = object.BFS()
    score += np.count_nonzero(bfs[:,3*3:])*0.001 #路の量で加算
    score += np.count_nonzero(bfs[:,3*5:])*0.001 #路の量で加算
    score += np.count_nonzero(bfs[:,3*8:])*0.001 #路の量で加算
    

    return score
def evalFunc(individual):
    #ネットワーク設定
    net = Network2()
    for i in range(4):
        net.fc1.weight[i] = torch.tensor(individual[0+i*8:8+i*8])
    net.fc2.weight[0] = torch.tensor(individual[32:36])
    b = nn.Parameter(torch.tensor(individual[36:]).to(device), requires_grad=False)
    net.fc1.bias = b

    
    GameObj = ojamamono(net=net_res, preprocess=preprocess)
    eval_score = 0
        
    #ゲームプレイ
    T = 5 #何ゲームプレイするか?
    t = 0 #経過ゲーム数
    turn = 0
    
    while t < T:
        GameObj.__init__(net=net_res, preprocess=preprocess) #ゲームリセット
        turn = 0
        
        hand0 = [GameObj.deck.pop() for i in range(4)]  #自分
        hand1 = [GameObj.deck.pop() for i in range(4)]  #自分のコピー
        hand2 = [GameObj.deck.pop() for i in range(4)]  #ランダムプレイ
        hand3 = [GameObj.deck.pop() for i in range(4)]  #ランダムプレイ
        hand4 = [GameObj.deck.pop() for i in range(4)]  #お邪魔もの
        hands = [hand0, hand1, hand2, hand3, hand4]
        Player_order = [0, 1, 2, 3, 4]
        random.shuffle(Player_order)
        order_turn = 0
        
        #ゲーム終了フラグ
        passCount = 0
        
        #main loop
        while True:
            turn += 1
            now_player = Player_order[order_turn]   #現在のプレイヤー
            candidate = GameObj.GetCandidateList(hands[now_player]) #行動を列挙
            
            """自分とコピーの手番"""
            if now_player == 0 or now_player == 1:    #自分
                
                #ゲームの終了処理
                if (len(candidate) == 0 or len(hands[now_player]) == 0):   #候補が存在しない or 手札が0
                    if len(GameObj.deck) == 0:  #デッキが0
                        passCount += 1
                    elif len(hands[now_player]) != 0:   #山札があり手札が残っていれば捨てて引く
                        random.shuffle(hands[now_player])
                        hands[now_player].pop()
                        hands[now_player].append(GameObj.deck.pop())
                    else:                               #山札があり手札がなければ引く (このような状況は想定されない)
                        hands[now_player].append(GameObj.deck.pop())
                        print("ERROR : hands")
                        pass
                
                else:
                    #一般のゲーム処理
                    passCount = 0
                    pull = 2 #カードを引く枚数
                    
                    InputList = [np.array(score) for score in map(GameObj.GetBoardScore, candidate)]
                    evalList = [e[0].item() for e in map(net, InputList)] #評価値を入手
                    
                    best_move = candidate[np.argmax(evalList)]      #最善手
                    #worst_move = candidate[np.argmin(evalList)]     #再悪手
                    y = best_move[0]
                    x = best_move[1]
                    c = best_move[2]
                    d = best_move[3]
                    
                    hands[now_player].remove(c)
                    
                    GameObj.SetCard(y, x, c, d)  #カードを設置
                    
                    if len(GameObj.deck) != 0:
                        hands[now_player].append(GameObj.deck.pop())
                
            """ランダムプレイ"""
            if now_player == 2:
                #ゲームの終了処理
                if (len(candidate) == 0 or len(hands[now_player]) == 0):   #候補が存在しない or 手札が0
                    if len(GameObj.deck) == 0:  #デッキが0
                        passCount += 1
                    elif len(hands[now_player]) != 0:   #山札があり手札が残っていれば捨てて引く
                        random.shuffle(hands[now_player])
                        hands[now_player].pop()
                        hands[now_player].append(GameObj.deck.pop())
                    else:                               #山札があり手札がなければ引く (このような状況は想定されない)
                        hands[now_player].append(GameObj.deck.pop())
                        print("ERROR : hands")
                        pass
                
                else:
                    #ランダムプレイ
                    passCount = 0
                    random.shuffle(candidate)
                    y = candidate[0][0]
                    x = candidate[0][1]
                    c = candidate[0][2]
                    d = candidate[0][3]
                    hands[now_player].remove(c)
                    GameObj.SetCard(y, x, c, d)  #カードを設置
                    if len(GameObj.deck) != 0:
                        hands[now_player].append(GameObj.deck.pop()) #カードを引く
                        
            """お邪魔もの"""
            if now_player == 3 or now_player == 4:    #お邪魔もの処理
                #ゲームの終了処理
                if (len(candidate) == 0 or len(hands[now_player]) == 0):   #候補が存在しない or 手札が0
                    if len(GameObj.deck) == 0:  #デッキが0
                        passCount += 1
                    elif len(hands[now_player]) != 0:   #山札があり手札が残っていれば捨てて引く
                        random.shuffle(hands[now_player])
                        hands[now_player].pop()
                        hands[now_player].append(GameObj.deck.pop())
                    else:                               #山札があり手札がなければ引く (このような状況は想定されない)
                        hands[now_player].append(GameObj.deck.pop())
                        print("ERROR : hands")
                        pass
                else:
                    max_manhattan = 0
                    move = []
                    for c in candidate:
                        g = copy.deepcopy(GameObj.goal)
                        now_sc = GameObj.GetBoardScore(c, np.array(g)-22, none_nn=True)
                        if now_sc[0] > max_manhattan:  #最小マンハッタン距離を長くできる
                            move = c
                            max_manhattan = now_sc[0]
                    if move == []:
                        hands[now_player].remove(candidate[0][2]) 
                        print(f"candidate{candidate}\n move{move}")
                    else:
                        hands[now_player].remove(move[2])   
                        GameObj.SetCard(move[0], move[1], move[2], move[3])  #カードを設置
                    if len(GameObj.deck) != 0:
                        hands[now_player].append(GameObj.deck.pop())       

            """共通処理"""
            #ゲーム終了のチェック
            if GameObj.GoalCheck() != 0 or 4 < passCount: 
                if passCount == 0 and Player_order[order_turn] == 0:
                    eval_score += 0.1   #自分がクリアする
                eval_score += reward(GameObj) - t*0.01 #tは経過ターン数 ターンがかかりすぎるとマイナス発生
                t+=1
                break
            
            #プレイヤーの切り替え
            order_turn += 1
            if order_turn == 5:
                order_turn = 0
            
    #T回プレイを行い平均をスコアとする
    return eval_score/T,

    
# 適応度クラスの作成
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 個体クラスの作成
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolboxの作成
toolbox = base.Toolbox()


# 遺伝子を生成する関数"attr_gene"を登録
toolbox.register("attr_gene", random.uniform, MIN_GEN_VAL, MAX_GEN_VAL)

# 個体を生成する関数”individual"を登録
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_gene, GEN_LENGTH)

# 個体集団を生成する関数"population"を登録
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 評価関数"evaluate"を登録
toolbox.register("evaluate", evalFunc)

# 交叉を行う関数"mate"を登録
toolbox.register("mate", tools.cxBlend, alpha=0.2)

# 変異を行う関数"mutate"を登録
toolbox.register("mutate", tools.mutGaussian, mu=[
                 MU for i in range(GEN_LENGTH)], sigma=[SIGMA for j in range(GEN_LENGTH)], indpb=0.2)

# 個体選択法"select"を登録
toolbox.register("select", tools.selTournament, tournsize=10)


if __name__ == '__main__':
    #並列処理用
    pool = multiprocessing.Pool(20)
    toolbox.register("map", pool.map)

    # 個体集団の生成
    pop = toolbox.population(n=POP_SIZE)    #金鉱堀
    print("Start of evolution")
    
    # 個体集団の適応度の評価
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(pop))

    # 適応度の抽出
    fits = [ind.fitness.values[0] for ind in pop]
    
    # 進化ループ開始
    g = 0
    
    #ログの設定
    with open('log.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Min", "Max", "Ave", "Std", "Best individual"])


    while g < N_GEN:

        g = g + 1
        print("-- Generation %i --" % g)

        # 次世代個体の選択・複製
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):     #金鉱堀
            # 交叉させる個体を選択
            if random.random() < CX_PB:
                toolbox.mate(child1, child2)

                # 交叉させた個体は適応度を削除する
                del child1.fitness.values
                del child2.fitness.values

        # 変異
        for mutant in offspring:

            # 変異させる個体を選択
            if random.random() < MUT_PB:
                toolbox.mutate(mutant)

                # 変異させた個体は適応度を削除する
                del mutant.fitness.values

        # 適応度を削除した個体について適応度の再評価を行う (金鉱堀)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))

        # 個体集団を新世代個体集団で更新
        pop[:] = offspring

        # 新世代の全個体の適応度の抽出
        fits = [ind.fitness.values[0] for ind in pop]
        
        # 適応度の統計情報の出力
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        #ログ出力
        with open('log.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([g, min(fits), max(fits), mean, std, tools.selBest(pop, 1)[0]])
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(f"-- Generation {g} --\n")
            f.write(f"Min {min(fits)}\n")
            f.write(f"Max {max(fits)}\n")
            f.write(f"Ave {mean}\n")
            f.write(f"Std {std}\n")
            f.write(f"Best individual    {tools.selBest(pop, 1)[0]}\n")

    print("-- End of (successful) evolution --")

    # 最良個体の抽出
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))