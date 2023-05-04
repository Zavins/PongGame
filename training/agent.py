import random

async def receive_state(websocket, data):
    position = random.choice([-1, 0, 1])
    angle = random.choice([-1, 0, 1])
    return {"position": position, "angle": angle}