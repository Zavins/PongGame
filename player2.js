const player2 = {
    x: null,
    y: null,
    angle: 0,
    slope: 0,
    yintercept: 0,
    plat: -4,
    score: 0,
    hit: 0,
    name: "Player 2",
    update: function () {
        this.slope = getSlopeByAngle(this.angle);
        this.yintercept = getYintercept(this.x, this.y + this.plat, this.slope)
    },
    draw: function () {
        table.save();
        table.beginPath()
        table.translate(this.x, this.y);
        table.rotate(this.angle * (Math.PI / 180));
        table.fillStyle = 'rgb(0, 200, 0)';
        table.fillRect(-20, -4, 40, 8.0);
        table.restore();
    },

    init: function () {
        this.x = tWidth / 2;
        this.y = tHeight - 8 - 8;
        this.angle = 0;
        this.slope = 0;
        this.yintercept = 0;
        this.plat = -4;
        this.score = 0;
        this.name = "Player 2";
        this.hit = 0;
    },
}