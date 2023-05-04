const player1 = {
    x: null,
    y: null,
    angle: 0,
    slope: 0,
    yintercept: 0,
    plat: 4,
    score: 0,
    hit: 0,
    name: "PongMaster *.*",
    update: function () {
        this.slope = getSlopeByAngle(this.angle);
        this.yintercept = getYintercept(this.x, this.y + this.plat, this.slope)
    },

    draw: function () {
        table.save();
        table.translate(this.x, this.y);
        table.rotate(this.angle * (Math.PI / 180));
        table.fillStyle = 'rgb(0, 200, 0)';
        table.fillRect(-20, -4, 40, 8.0);
        table.restore();
    },

    moveAI: function () {
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

    init: function () {
        this.x = tWidth / 2;
        this.y = 16;
        this.angle = 0;
        this.slope = 0;
        this.yintercept = 0;
        this.plat = 4;
        this.score = 0;
        this.name = "PongMaster *.*";
        this.hit = 0;
    }
}