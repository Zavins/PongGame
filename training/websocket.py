import traceback
import random
from logging import getLogger

from fastapi import (APIRouter, Depends, FastAPI, HTTPException, WebSocket,
                     WebSocketDisconnect, status)
from fastapi.websockets import WebSocketState
from agent import receive_state

log = getLogger(__name__)

game = APIRouter()


@game.websocket("/") # type: ignore
async def chat(websocket: WebSocket):
    await websocket.accept()
    while websocket.client_state == WebSocketState.CONNECTED:
        try:
            data = await websocket.receive_json()
            response = await receive_state(data)
            #process the data then send the action.
            await websocket.send_json(response)
        except WebSocketDisconnect as e:
            log.info(f"Disconnected")
            return
        except Exception as e:
            log.error(f"error: {traceback.format_exc()}")
            return

app = FastAPI()
app.include_router(game)
