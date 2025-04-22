from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import random
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse, FileResponse

import connect4

app = FastAPI()

# Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

templates = Jinja2Templates(directory="templates")


class GameBoard(BaseModel):
    board: list[list[int]]


class AIMoveRequest(BaseModel):
    board: list[list[int]]
    player: int
    engine: str
    depth: int


def convert_board_to_bitmap(board, player) -> connect4.GameState:
    game = connect4.GameState()
    game.red = 0
    game.yellow = 0
    game.player = connect4.Player.RED if player == 1 else connect4.Player.YELLOW

    for row in range(6):
        for col in range(7):
            if board[row][col] == 1:
                game.red |= 1 << (row * 7 + col)
            elif board[row][col] == 2:
                game.yellow |= 1 << (row * 7 + col)

    return game


@app.post("/ai_move")
async def get_ai_move(request: AIMoveRequest):
    """Calculate AI move based on the selected engine"""
    board = request.board
    player = request.player
    engine = request.engine
    depth = request.depth

    game = convert_board_to_bitmap(board, player)
    score, column, tree = connect4.play(game, engine, depth)

    return {"column": column, "score": score}


@app.post("/get_score")
async def get_score(request: GameBoard):
    """Get the score for the given game settings"""
    board = request.board
    game = convert_board_to_bitmap(board, 1)

    return {"score": (game.evaluate_final())}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return templates.TemplateResponse("index.html", {"request": {}, })


@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse("/assets/favicon.svg")


app.mount("/", StaticFiles(directory="./static"), name="static")

# Mount static files (CSS, JS)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
