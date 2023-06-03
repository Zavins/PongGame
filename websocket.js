const socket = new WebSocket("ws://localhost:8080");

function sendGameData(player1, player2, ball) {
    if (socket.readyState === 1) {
        socket.send(JSON.stringify({
            player1: {
                x: player1.x,
                angle: player1.angle,
                score: player1.score,
                hit: player1.hit,
            },
            player2: {
                x: player2.x,
                angle: player2.angle,
                score: player2.score,
                hit: player2.hit,
            },
            ball: {
                x: ball.x,
                y: ball.y,
                radius: ball.radius,
                speed: ball.speed,
                serve: ball.serve,
                velocity: { x: ball.velocity.x, y: ball.velocity.y }, //x y
            }
        }));
    }

}
// Connection opened
socket.addEventListener("open", (event) => {
    console.log("socket connected")
});

// Listen for messages
socket.addEventListener("message", (event) => {
    gamedata = makeAction(event.data);
    sendGameData(gamedata[0], gamedata[1], gamedata[2])
});