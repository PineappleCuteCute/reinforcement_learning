# simulation.py

import pygame
from environment import Environment
import csv

# Constants for the screen size
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

def draw_environment(screen, env, trail):
    """Render the environment using Pygame."""
    screen.fill(WHITE)  # Clear screen with white background

    # Draw trail
    for point in trail:
        pygame.draw.circle(screen, GREEN, (int(point[0]), int(point[1])), 3)

    # Draw robot
    robot_pos = env.robot.position
    pygame.draw.circle(screen, BLUE, (int(robot_pos[0]), int(robot_pos[1])), env.robot.size)

    # Draw obstacles
    for obs in env.obstacles:
        obs_pos = obs["position"]
        obs_size = obs["size"]
        pygame.draw.circle(screen, RED, (int(obs_pos[0]), int(obs_pos[1])), obs_size)

def save_robot_data(trail, filename="robot_data.csv"):
    """Save robot trail data to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y"])  # Header
        writer.writerows(trail)

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Simulation with Trail")
    clock = pygame.time.Clock()

    # Initialize environment
    env = Environment(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, num_obstacles=5)
    trail = []  # To store robot positions

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Example action: Move diagonally (change x, y velocity)
        action = [0.5, -0.3]
        state, reward, done = env.step(action)

        # Record robot position
        trail.append(tuple(env.robot.position))

        # Draw the environment
        draw_environment(screen, env, trail)
        pygame.display.flip()

        # End simulation if collision happens
        if done:
            print("Collision detected! Simulation stopped.")
            break

        # Limit frame rate
        clock.tick(FPS)

    # Save trail data
    save_robot_data(trail)
    print("Robot data saved to 'robot_data.csv'.")

    pygame.quit()

if __name__ == "__main__":
    main()
