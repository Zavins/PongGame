var cv;
var tWidth = 250; // table width and height
var tHeight = 450;
var loop;

function main() {
    document.getElementById('button').innerHTML = "Restart Game";
    disposeGame();
    initGame();
    player2.name = getPlayerName();
    ball.speed = document.getElementById('level').selectedIndex * 0.5 + 2;
    updateGame();
    // if (!loop) {
    //     loop = function () {
    //         updateGame();
    //         window.requestAnimationFrame(loop, cv); //continue loop 
    //     }
    //     window.requestAnimationFrame(loop, cv); // first loop
    // }
}

function disposeGame() {
    document.removeEventListener("mousemove", function () { });
}

function initGame() {
    //get Ping Pong Table
    cv = document.getElementById('table');
    cv.width = tWidth;
    cv.height = tHeight;
    //init table
    table = cv.getContext('2d');
    table.clearRect(0, 0, tWidth, tHeight);
    table.fillStyle = 'rgba(0, 33, 84, 1.0)'; // table color
    table.fillRect(0, 0, tWidth, tHeight);
    table.fillStyle = 'rgba(255, 255, 255, 0.8)';
    table.fillRect(0, tHeight / 2 - 4, tWidth, 8.0);
    //init balls
    ball.init();
    ball.reset();
    //init player1
    player1.init();
    //init player2
    player2.init();
    //disable mouse
    // document.addEventListener('mousemove', (e) => {
    //     var mousePos = getMousePos(cv, e);
    //     //console.log(e);
    //     player2.x = clampNumber(mousePos.x, 20, tWidth - 20);
    //     player2.angle = clampNumber(mousePos.y - tHeight / 2, -45, 45);
    //     //player1.angle = 0;
    // });
}

function makeAction(data){
    data = JSON.parse(data)
    player2.action(data["position"], data["angle"]);
    updateGame()
}

function updateGame() {
    clearTable();
    drawTable();
    // checkScore(); Run infinitely (for training)
    ball.update();
    ball.draw();
    player1.moveAI();
    player1.update();
    player1.draw();
    player2.update();
    player2.draw();
    sendGameData(player1, player2, ball);
}

function drawTable() {
    table.save();
    table.clearRect(0, 0, tWidth, tHeight);
    table.fillStyle = 'rgba(0, 33, 84, 1.0)'; // table color
    table.fillRect(0, 0, tWidth, tHeight);
    table.fillStyle = 'rgba(255, 255, 255, 0.8)';
    table.fillRect(0, tHeight / 2 - 4, tWidth, 8.0);
    table.fillStyle = 'rgb(200,200,0)';
    table.textAlign = "left";
    table.font = "12px Arial";
    table.fillText(player1.name.toString(), 2, 12);
    table.font = "30px Comic Sans MS";
    table.fillText(player1.score.toString(), 10, 40);
    table.textAlign = "right";
    table.font = "12px Arial";
    table.fillText(player2.name.toString(), tWidth - 2, tHeight - 6);
    table.font = "30px Comic Sans MS";
    table.fillText(player2.score.toString(), tWidth - 10, tHeight - 20);
    table.restore();
}

function clearTable() {
    table.clearRect(0, 0, tWidth, tHeight);
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function getSlopeByAngle(angle) {
    return Math.tan((angle) / 180 * Math.PI);
}

function getAngleBySlope(slope) {
    return Math.atan(slope) / Math.PI * 180;
}

function getYintercept(x, y, slope) {
    return y - slope * x;
}

function getDistance(ball, player) {
    var x;
    var y;
    if (player.slope !== 0) {
        var bslope = -1 / player.slope;
        //console.log(bslope,player.slope);
        var yintercept = getYintercept(ball.x, ball.y, bslope);
        //console.log(player.slope,player.x,player.yintercept);
        x = (player.yintercept - yintercept) / (bslope - player.slope);
        //console.log(Math.cos(player.angle/180*Math.PI)*40);
        //console.log(player.x)
        x = clampNumber(x, player.x - Math.cos(player.angle / 180 * Math.PI) * 20, player.x + Math.cos(player.angle / 180 * Math.PI) * 20);
        //console.log(x);
        y = x * player.slope + player.yintercept;
        //console.log(player.x, x);
    } else {
        x = ball.x;
        x = clampNumber(x, player.x - Math.cos(player.angle / 180 * Math.PI) * 20, player.x + Math.cos(player.angle / 180 * Math.PI) * 20);
        y = player.y + player.plat;
    }
    var distance = Math.sqrt(Math.pow((x - ball.x), 2) + Math.pow((y - ball.y), 2));
    //console.log(distance);
    return distance
}

function getPlayerName() {
    var name = prompt("Please enter your name", "Player 2");
    return name == null ? "player 2" : name;
}

function clampNumber(num, min, max) {
    return num <= min ? min : num >= max ? max : num;
}

function checkScore() {
    if (player1.score >= 5) {
        window.alert("[" + player1.name + "]" + " defeats " + "[" + player2.name + "]" + " by " + (player1.score - player2.score).toString() + " points!\n\n" +
            "Final Score: " + player1.score.toString() + " : " + player2.score.toString());
        main();
    }
    if (player2.score >= 5) {
        window.alert("[" + player2.name + "]" + " defeats " + "[" + player1.name + "]" + " by " + (player2.score - player1.score).toString() + " points!\n\n" +
            "Final Score: " + player2.score.toString() + " : " + player1.score.toString());
        main();
    }
} 