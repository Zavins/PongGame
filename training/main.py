import uvicorn
import tensorflow as tf
import GPUtil

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        print("Using GPU for TensorFlow computations.")
    else:
        print("No compatible GPU found. TensorFlow will use CPU for computations.")
        GPUtil.getAvailable()
    uvicorn.run(
        "websocket:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
        use_colors=True,
        proxy_headers=True,
    )
