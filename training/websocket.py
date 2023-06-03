import traceback
import random
from logging import getLogger

from fastapi import (APIRouter, Depends, FastAPI, HTTPException, WebSocket,
                     WebSocketDisconnect, status)
from fastapi.websockets import WebSocketState
from agent import DQN

log = getLogger(__name__)

game = APIRouter()


@game.websocket("/") # type: ignore
async def socket(websocket: WebSocket):

    await websocket.accept()
    
    while websocket.client_state == WebSocketState.CONNECTED:
        try:
            async def game_step(response):
                await websocket.send_json(response)
                data = await websocket.receive_json()
                return data
            dqn = DQN(game_step)
            await dqn.train()
        except WebSocketDisconnect as e:
            log.info(f"Disconnected")
            return
        except Exception as e:
            log.error(f"error: {traceback.format_exc()}")
            return

app = FastAPI()
app.include_router(game)
