import uvicorn
import torch
print(torch.cuda.is_available())
print(torch.zeros(1).cuda())

if __name__ == "__main__":
    uvicorn.run(
        "websocket:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
        use_colors=True,
        proxy_headers=True,
    )
