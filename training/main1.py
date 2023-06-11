import uvicorn

if __name__ == "__main__":

    uvicorn.run(
        "websocket1:app",
        host="0.0.0.0",
        port=8081,
        log_level="info",
        access_log=True,
        use_colors=True,
        proxy_headers=True,
    )
