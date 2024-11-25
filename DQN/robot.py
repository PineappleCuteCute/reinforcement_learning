class Robot:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.velocity_x = 0
        self.velocity_y = 0

    def move(self, action):
        self.x += action[0]
        self.y += action[1]
        self.x = max(self.size, min(self.x, 800 - self.size))
        self.y = max(self.size, min(self.y, 600 - self.size))

    def set_velocity(self, velocity_x, velocity_y):
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def get_position(self):
        return [self.x, self.y]
