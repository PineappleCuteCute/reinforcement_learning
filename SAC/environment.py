import numpy as np

class Environment:
    def __init__(self, goal_position, obstacles):
        self.goal_position = np.array(goal_position)  # Vị trí đích
        self.obstacles = obstacles  # Danh sách các chướng ngại vật

    def check_collision(self, position):
        # Kiểm tra nếu robot có va chạm với chướng ngại vật
        for obs in self.obstacles:
            if np.array_equal(position, np.array(obs)):
                return True
        return False
    
    def step(self, action, robot):
        # Thực hiện hành động và tính toán phần thưởng
        robot.move(action)
        
        # Kiểm tra va chạm với chướng ngại vật
        if self.check_collision(robot.get_position()):
            return robot.get_position(), -10, False  # Phạt khi va chạm

        # Kiểm tra nếu robot đến đích
        if np.array_equal(robot.get_position(), self.goal_position):
            return robot.get_position(), 10, True  # Phần thưởng khi đến đích

        # Trả lại phần thưởng bình thường
        return robot.get_position(), -1, False

    def reset(self, robot):
        # Đặt lại robot và môi trường về trạng thái ban đầu
        robot.reset()
        return robot.get_position()
