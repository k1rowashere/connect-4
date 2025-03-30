
rows = 6
columns = 7

AiPlayer = '2'
HumanPlayer = '1'
class Move:
    def __init__(self, row, column):
        self.column = column 
        self.row = row    

    
    def __repr__(self):
        return f"Move(row={self.row}, col={self.column})"
    
def getValidMoves(board):
    validMoves = []
    for c in range(columns):
        for r in range(rows-1,-1,-1):
            if board[r *columns + c] == '0':
                move = Move(r,c)
                validMoves.append(move)
                break # lowest row go to the next column
    return validMoves

def makeMove(board,move,player): #rows= 7
    boardList = list(board)
    index = move.row * columns + move.column
    boardList[index] = str(player)
    return ''.join(boardList)


def printBoard(board):
    for r in range(rows):
        row = board[r * columns : (r + 1) * columns]
        print(' '.join(row))
    print()  # empty line after board

def checkWin(board,player):
    score = 0
    target = 0
    target = player * 4
    #horizonatl
    for r in range (rows):
        for c in range(columns - 4):
            start = r * columns + c
            end = start + 4 
            if board[start:end] == target:
                score +=1 
    #vertical
    for r in range(rows - 3):
        for c in range(columns):
            index1 = r * columns + c
            index2 = (r + 1) * columns + c
            index3 = (r + 2) * columns + c
            index4 = (r + 3) * columns + c
            if board[index1] == board[index2] == board[index3] == board[index4] == player:
                score +=1

    # diagonal down-right
    for r in range(rows - 3):
        for c in range(columns - 3):
            index1 = r * columns + c
            index2 = (r + 1) * columns + (c + 1)
            index3 = (r + 2) * columns + (c + 2)
            index4 = (r + 3) * columns + (c + 3)
            if board[index1] == board[index2] == board[index3] == board[index4] == player:
                score +=1

    # diagonal down-left
    for r in range(rows - 3):
        for c in range(3, columns):
            index1 = r * columns + c
            index2 = (r + 1) * columns + (c - 1)
            index3 = (r + 2) * columns + (c - 2)
            index4 = (r + 3) * columns + (c - 3)
            if board[index1] == board[index2] == board[index3] == board[index4] == player:
                score +=1
    
    return score 


def minimax(board, depth, maxPlayer, space=0):
    spaceStr = "-- " * (2 * space)
    if '0' not in board or depth == 0: # gameEneded or depth = 0
        if checkWin(board, AiPlayer):
            return 1000, None   #  positive score if AI won
        elif checkWin(board, HumanPlayer):
            return -1000, None  #  negative score if Human won
        else:
            # If depth limit reached, evaluate
            return 5,None
    
    validMoves = getValidMoves(board)
    bestMove = None
    if maxPlayer:
        maxValue =  float('-inf')
        print(spaceStr + "Maximize (AI)")
        for move in validMoves:
            newBoard = makeMove(board, move, AiPlayer)
            print(spaceStr + f"Max Trying Move: {move}")
            printBoard(newBoard)
            newValue, _ = minimax(newBoard, depth - 1, False, space + 1)  # minimize next turn
            if newValue > maxValue:
                maxValue = newValue
                bestMove = move
        return maxValue, bestMove
    else:
        minValue =  float('inf')
        print(spaceStr + "Minimize (human)")
        for move in validMoves:
            newBoard = makeMove(board, move, HumanPlayer)
            print(spaceStr + f"Min Trying Move: {move}")
            #printBoard(newBoard)
            newValue, _ = minimax(newBoard, depth - 1, True,space + 1)  # maximize next turn
            if newValue > minValue:
                minValue = newValue
                bestMove = move
        return minValue, bestMove
        

if __name__ == "__main__":

    board = "0" * ( rows * columns)
    while '0' in board:
        #human turn
        print("Human Turn")
        while True:
            moveCol = int(input("Enter column (0-6): "))
            moveRow = int(input("Enter row (0-5): "))
            move = Move(moveRow,moveCol)
            if any(m.row == move.row and m.column == move.column for m in getValidMoves(board)):
                break
            else :
                print("Invalid move ... Renter move")

        board = makeMove(board, Move(moveRow,moveCol),'1')
        printBoard(board)

        #AI turn
        print("AI turn")
        score, bestMove = minimax(board, 2, True)
        board = makeMove(board, bestMove,'2')
        print("Move", bestMove)
        printBoard(board)

# 0
# 7
