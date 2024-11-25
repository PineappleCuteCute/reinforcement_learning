# File: robot.py

class Robot:
    def __init__(self, x, y, size=20):
        self.position = [x, y]
        self.size = size
        self.velocity = [0, 0]

    def perform_action(self, action):
        """
        action[0]: Change in velocity x
        action[1]: Change in velocity y
        """
        self.velocity[0] += action[0]
        self.velocity[1] += action[1]
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Keep robot within bounds
        self.position[0] = max(0, min(self.position[0], 800))  # Assuming width=800
        self.position[1] = max(0, min(self.position[1], 600))  # Assuming height=600

    def get_state(self):
        """Return current state."""
        return {
            "position": self.position,
            "velocity": self.velocity
        }
