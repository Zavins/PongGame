var cv;
var tWidth = 250; // table width and height
var tHeight = 450;
var loop;
var ball = {
 x: null,
 y: null,
 radius: 8,
 speed: 2,
 serve: 0,
 velocity: {
  x: 0,
  y: 0
 }, //x y
 update: function() {
  ball.x += this.velocity.x * this.speed;
  ball.y += this.velocity.y * this.speed;
  if (this.x - this.radius <= 0 || this.x + this.radius >= tWidth) {
   if (this.x <= 0) this.reset();
   this.velocity.x *= -1;
  }
  if (this.y - this.radius <= 0) {
   player2.score++;
   this.reset();
   //this.velocity.y *= -1;
  } else if (this.y + this.radius >= tHeight) {
   player1.score++;
   this.reset();
  }
  //TODO
  //find equation for line btw ball and paddle 2
  var distance = getDistance(this, player1);
  var angle = getAngleBySlope(this.velocity.y / this.velocity.x);
  if (distance <= this.radius) {
   var pAngle = player1.angle;
   //console.log(angle,pAngle);
   var newAngle = 0;
   if (angle > 0 && pAngle > 0) {
    newAngle = -180 + Math.abs(angle) - 2 * Math.abs(pAngle);
   } else if (angle > 0 && pAngle <= 0) {
    newAngle = -180 + Math.abs(angle) + 2 * Math.abs(pAngle);
   } else if (angle < 0 && pAngle >= 0) {
    newAngle = -(2 * Math.abs(pAngle) + Math.abs(angle));
   } else if (angle < 0 && pAngle < 0) {
    newAngle = 2 * Math.abs(pAngle) - Math.abs(angle);
   } else if (angle === 0) {
    newAngle = -90
   }
   var newSlope = getSlopeByAngle(newAngle);
   //console.log(angle,newAngle);
   //console.log(angle,newAngle,newSlope);
   this.velocity.x = Math.sqrt(Math.pow(this.speed, 2) / (1 + Math.pow(newSlope, 2)));
   this.velocity.y = this.velocity.x * newSlope;
   if (newAngle < -90) {
    this.velocity.x *= -1;
   } else if (newAngle2 === 90 || newAngle2 === -90) {
    this.velocity.x = 0;
   } else {
    this.velocity.y *= -1;
   }
   //console.log(this.velocity.x, this.velocity.y)
  }
  var distance2 = getDistance(this, player2);
  if (distance2 <= this.radius) {
   var pAngle2 = player2.angle;
   var newAngle2 = 0;
   if (angle > 0 && pAngle2 > 0) {
    newAngle2 = Math.abs(angle) - 2 * Math.abs(pAngle2);
   } else if (angle > 0 && pAngle2 <= 0) {
    newAngle2 = Math.abs(angle) + 2 * Math.abs(pAngle2);
   } else if (angle < 0 && pAngle2 > 0) {
    newAngle2 = 180 - Math.abs(angle) - 2 * Math.abs(pAngle2);
   } else if (angle < 0 && pAngle2 <= 0) {
    newAngle2 = 180 - Math.abs(angle) + 2 * Math.abs(pAngle2);
   } else if (angle === 0) {
    newAngle2 = 90
   }
   var newSlope2 = getSlopeByAngle(newAngle2);
   //console.log(angle,newAngle2);
   //console.log(angle,newAngle,newSlope);
   this.velocity.x = Math.sqrt(Math.pow(this.speed, 2) / (1 + Math.pow(newSlope2, 2)));
   this.velocity.y = this.velocity.x * newSlope2;
   if (newAngle2 > 90) {
    this.velocity.x *= -1;
   } else if (newAngle2 === 90 || newAngle2 === -90) {
    this.velocity.x = 0;
    this.velocity.y *= -1;
   } else {
    this.velocity.y *= -1;
   }
   //console.log(this.velocity.x, this.velocity.y)
  }
  //console.log(this.speed);
 },
 draw: function() {
  table.beginPath()
  table.arc(this.x, this.y, this.radius, 0, 2 * Math.PI, false);
  table.fillStyle = 'rgb(255, 255, 255)';
  table.fill();
 },
 reset: function() {
  this.x = tWidth / 2;
  this.y = tHeight / 2;
  this.velocity.x = 0;
  if (this.serve === 0) {
   this.serve = Math.floor(Math.random() * 2 + 1);
  }
  if (this.serve === 1) {
   this.serve = 2;
   this.velocity.y = -1;
  } else if (this.serve === 2) {
   this.serve = 1;
   this.velocity.y = 1;
  }
 },
 init: function() {
  this.x = null;
  this.y = null;
  this.radius = 8;
  this.speed = 2;
  this.serve = 0;
  this.velocity = {
   x: 0,
   y: 0
  };
 }
}

var player1 = {
 x: null,
 y: null,
 angle: 0,
 slope: 0,
 yintercept: 0,
 plat: 4,
 score: 0,
 name: "PongMaster *.*",
 update: function() {
  this.slope = getSlopeByAngle(this.angle);
  this.yintercept = getYintercept(this.x, this.y + this.plat, this.slope)
 },
 draw: function() {
  table.save();
  table.translate(this.x, this.y);
  table.rotate(this.angle * (Math.PI / 180));
  table.fillStyle = 'rgb(0, 200, 0)';
  table.fillRect(-20, -4, 40, 8.0);
  table.restore();
 },
 moveAI: function() {
  if (ball.y <= tHeight / 3 && ball.velocity.y < 0) {
   //move to ball.x
   if (this.x - ball.x > 6) {
    this.x -= 0.8 * ball.speed * ball.speed;
   } else if (this.x - ball.x < -6) {
    this.x += 0.8 * ball.speed * ball.speed;
   } else {
    this.x = clampNumber(ball.x, 0, tWidth);
   }
   this.x = clampNumber(this.x, 20, tWidth - 20);
   var pAngle = -getAngleBySlope(ball.velocity.y / ball.velocity.x) / 3;
   if (this.angle < pAngle - 5) {
    this.angle += 0.5 * ball.speed * ball.speed;
   } else if (this.angle > pAngle + 5) {
    this.angle -= 0.5 * ball.speed * ball.speed;
   } else {
    this.angle = clampNumber(pAngle, -45, 45);
   }
  }
 },
 init: function() {
  this.x = tWidth / 2;
  this.y = 16;
  this.angle = 0;
  this.slope = 0;
  this.yintercept = 0;
  this.plat = 4;
  this.score = 0;
  this.name = "PongMaster *.*";
 }
}

var player2 = {
 x: null,
 y: null,
 angle: 0,
 slope: 0,
 yintercept: 0,
 plat: -4,
 score: 0,
 name: "Player 2",
 update: function() {
  //this.x = clampNumber(this.x, 20, tWidth - 20);
  //this.angle = clampNumber(this.angle,-45,45);
  this.slope = getSlopeByAngle(this.angle);
  this.yintercept = getYintercept(this.x, this.y + this.plat, this.slope)
  //console.log(this.slope*this.x+this.yintercept,this.y);

 },
 draw: function() {
  table.save();
  table.beginPath()
  table.translate(this.x, this.y);
  table.rotate(this.angle * (Math.PI / 180));
  table.fillStyle = 'rgb(0, 200, 0)';
  table.fillRect(-20, -4, 40, 8.0);
  table.restore();
 },
 init: function() {
  this.x = tWidth / 2;
  this.y = tHeight - 8 - 8;
  this.angle = 0;
  this.slope = 0;
  this.yintercept = 0;
  this.plat = -4;
  this.score = 0;
  this.name = "Player 2";
 }
}

function main() {
 document.getElementById('button').innerHTML = "Restart Game";
 disposeGame();
 initGame();
 player2.name = getPlayerName();
 ball.speed = document.getElementById('level').selectedIndex * 0.5 + 2;
 if (!loop) {
  loop = function() {
   updateGame();
   window.requestAnimationFrame(loop, cv); //continue loop 
  }
  window.requestAnimationFrame(loop, cv); // first loop
 }
}

function disposeGame() {
 //cv.clearRect(0, 0, cv.width, cv.height);
 document.removeEventListener("mousemove", function() {});
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
 document.addEventListener('mousemove', (e) => {
  var mousePos = getMousePos(cv, e);
  //console.log(e);
  player2.x = clampNumber(mousePos.x, 20, tWidth - 20);
  player2.angle = clampNumber(mousePos.y - tHeight / 2, -45, 45);
  //player1.angle = 0;
 });
}

function updateGame() {
 //console.log(player1.y,player2.y);
 clearTable();
 drawTable();
 ball.update();
 ball.draw();
 //console.log("hey");
 player1.moveAI();
 player1.update();
 player1.draw();
 player2.update();
 player2.draw();
 checkScore();
 //clearTable();

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
 if (player1.score >= 5 || player2.score >= 5) {
  if (player1.score >= 5)
   window.alert("[" + player1.name + "]" + " defeats " + "[" + player2.name + "]" + " by " + (player1.score - player2.score).toString() + " points!\n\n" +
    "Final Score: " + player1.score.toString() + " : " + player2.score.toString());
  if (player2.score >= 5)
   window.alert("[" + player2.name + "]" + " defeats " + "[" + player1.name + "]" + " by " + (player2.score - player1.score).toString() + " points!\n\n" +
    "Final Score: " + player2.score.toString() + " : " + player1.score.toString());
  main();
 }
} 