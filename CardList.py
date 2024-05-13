
rconnect = [0 for i in range(31)]

for i in range(len(rconnect)):
	rconnect[i] = [0,0] 
rconnect[0][0] = [ 0,0,0,0 ] 
rconnect[0][1] = [ 0,0,0,0 ] 

rconnect[1][0] = [ 1,1,1,1 ]  # 十字路
rconnect[1][1] = [ 1,1,1,1 ]  # 十字路

rconnect[2][0] = [ 1,1,1,0 ]  # T字路横
rconnect[2][1] = [ 1,0,1,1 ]  # T字路横

rconnect[3][0] = [ 1,1,0,1 ]  # T字路縦
rconnect[3][1] = [ 0,1,1,1 ]  # T字路縦

rconnect[4][0] = [ 1,1,0,0 ]  # 曲がり１
rconnect[4][1] = [ 0,0,1,1 ]  # 曲がり１

rconnect[5][0] = [ 1,0,0,1 ]  # 曲がり２
rconnect[5][1] = [ 0,1,1,0 ]  # 曲がり２

rconnect[6][0] = [ 1,0,1,0 ]  # 直線横
rconnect[6][1] = [ 1,0,1,0 ]  # 直線横

rconnect[7][0] = [ 0,1,0,1 ]  # 直線縦
rconnect[7][1] = [ 0,1,0,1 ]  # 直線縦


rconnect[8][0] = [ 2,2,2,2 ]  # 十字止
rconnect[8][1] = [ 2,2,2,2 ]  # 十字止

rconnect[9][0] = [ 2,2,2,0 ]  # T字止横
rconnect[9][1] = [ 2,0,2,2 ]  # T字止横

rconnect[10][0] = [ 2,2,0,2 ]  # T字止縦
rconnect[10][1] = [ 0,2,2,2 ]  # T字止縦

rconnect[11][0] = [ 2,2,0,0 ]  # 曲止め１
rconnect[11][1] = [ 0,0,2,2 ]  # 曲止め１

rconnect[12][0] = [ 2,0,0,2 ]  # 曲止め２
rconnect[12][1] = [ 0,2,2,0 ]  # 曲止め２

rconnect[13][0] = [ 2,0,2,0 ]  # 直線止横
rconnect[13][1] = [ 2,0,2,0 ]  # 直線止横

rconnect[14][0] = [ 0,2,0,2 ]  # 直線止縦
rconnect[14][1] = [ 0,2,0,2 ]  # 直線止縦

rconnect[15][0] = [ 2,0,0,0 ]  # 袋小路横
rconnect[15][1] = [ 0,0,2,0 ]  # 袋小路横

rconnect[16][0] = [ 0,2,0,0 ]  # 袋小路縦
rconnect[16][1] = [ 0,0,0,2 ]  # 袋小路縦

rconnect[20][0] = [ 1,1,1,1 ]  # スタート

rconnect[21][0] = [ 1,1,0,0 ]  # 石１
rconnect[21][1] = [ 0,0,1,1 ]  # 石１

rconnect[22][0] = [ 1,0,0,1 ]  # 石２
rconnect[22][1] = [ 0,1,1,0 ]  # 石２

rconnect[23][0] = [ 1,1,1,1 ]  # 金塊
rconnect[23][1] = [ 1,1,1,1 ]  # 金塊
rconnect[30][0] = [ 1,1,1,1 ]  # ゴール裏
rconnect[30][1] = [ 1,1,1,1 ]  # ゴール裏