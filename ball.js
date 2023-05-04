const ball = {
    x: null,
    y: null,
    radius: 8,
    speed: 2,
    serve: 0,
    velocity: { x: 0, y: 0 }, //x y

    update: function () {
        ball.x += this.velocity.x * this.speed;
        ball.y += this.velocity.y * this.speed;
        if (this.x - this.radius <= 0 || this.x + this.radius >= tWidth) {
            //Should never happen. This ball is out of bound
            if (this.x <= 0) this.reset();
            this.velocity.x *= -1; //Revert the velocity back and restart
        }
        if (this.y - this.radius <= 0) {
            //player 2 scores
            player2.score++;
            this.reset();
            //this.velocity.y *= -1;
        } else if (this.y + this.radius >= tHeight) {
            //player 1 scores
            player1.score++;
            this.reset();
        }

        let angle = getAngleBySlope(this.velocity.y / this.velocity.x);

        //find equation for line btw ball and paddle 2
        let distance = getDistance(this, player1);

        //if player 1 hit,
        if (distance <= this.radius) {
            let pAngle = player1.angle;
            //console.log(angle,pAngle);
            let newAngle = 0;
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

            let newSlope = getSlopeByAngle(newAngle);
            this.velocity.x = Math.sqrt(Math.pow(this.speed, 2) / (1 + Math.pow(newSlope, 2)));
            this.velocity.y = this.velocity.x * newSlope;
            if (newAngle < -90) {
                this.velocity.x *= -1;
            } else if (newAngle2 === 90 || newAngle2 === -90) {
                this.velocity.x = 0;
            } else {
                this.velocity.y *= -1;
            }
            player1.hit ++;
        }

        let distance2 = getDistance(this, player2);

        //if player 2 hit,
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
            player2.hit ++;
        }
    },


    draw: function () {
        table.beginPath()
        table.arc(this.x, this.y, this.radius, 0, 2 * Math.PI, false);
        table.fillStyle = 'rgb(255, 255, 255)';
        table.fill();
    },

    reset: function () {
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
    
    init: function () {
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