from typing import List, Tuple, Optional
import torch

class GridWorldEnv:
    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ACTION_NAMES = ["up", "right", "down", "left"]

    def __init__(self, width=10, height=10, start=(0,0), goal=(9,9),
                 obstacles=None, waypoints=None, max_steps: Optional[int]=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles: set[Tuple[int,int]] = set(obstacles or [])
        self.waypoints: List[Tuple[int,int]] = waypoints or []
        self.visited_waypoints: set[Tuple[int,int]] = set()
        self.max_steps = max_steps
        self.state = self.start
        self.steps = 0

        # Reward parameters
        self.step_penalty = -1
        self.wall_penalty = -2
        self.obstacle_penalty = -5
        self.waypoint_reward = 20
        self.goal_reward = 50
        self.goal_before_waypoints_penalty = -5

    def reset(self, start=None, goal=None, obstacles=None, waypoints=None, max_steps=None):
        if start: self.start = start
        if goal: self.goal = goal
        if obstacles: self.obstacles = set(obstacles)
        if waypoints is not None: self.waypoints = waypoints
        self.visited_waypoints = set()
        self.state = self.start
        if max_steps is not None:
            self.max_steps = max_steps
        self.steps = 0
        return self.state

    def step(self, action: int):
        assert action in [0,1,2,3]
        dx, dy = GridWorldEnv.ACTIONS[action]
        return self.step_vector(dx, dy)

    def step_by_name(self, action_name: str):
        if action_name not in GridWorldEnv.ACTION_NAMES:
            raise ValueError(f"Invalid action_name {action_name}")
        idx = GridWorldEnv.ACTION_NAMES.index(action_name)
        return self.step(idx)

    def step_vector(self, dx: int, dy: int):
        x, y = self.state
        nx, ny = x + dx, y + dy

        reward = self.step_penalty
        done = False
        info = {}

        # Check bounds
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            nx, ny = x, y
            reward = self.wall_penalty
            info["event"] = "wall"
        # Check obstacle
        elif (nx, ny) in self.obstacles:
            nx, ny = x, y
            reward = self.obstacle_penalty
            info["event"] = "obstacle"
        else:
            self.state = (nx, ny)
            # Check waypoint
            if self.state in self.waypoints and self.state not in self.visited_waypoints:
                self.visited_waypoints.add(self.state)
                reward += self.waypoint_reward
                info["event"] = "waypoint"

        # Check goal
        if self.state == self.goal:
            if set(self.waypoints).issubset(self.visited_waypoints):
                reward += self.goal_reward
                done = True
                info["event"] = "goal"
            else:
                reward += self.goal_before_waypoints_penalty
                done = False   # not done if waypoints chưa hết
                info["event"] = "goal_before_waypoints"

        # Timeout
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            done = True
            info["event"] = "timeout"
        
        info["visited_waypoints"] = list(self.visited_waypoints)
        return self.state, reward, done, info

    def get_map(self):
        return {
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "goal": self.goal,
            "obstacles": list(self.obstacles),
            "waypoints": self.waypoints,
            "max_steps": self.max_steps
        }

    def render_ascii(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for ox, oy in self.obstacles:
            if 0 <= ox < self.width and 0 <= oy < self.height:
                grid[oy][ox] = "#"
        for wx, wy in self.waypoints:
            if 0 <= wx < self.width and 0 <= wy < self.height:
                grid[wy][wx] = "v" if (wx, wy) in self.visited_waypoints else "W"
        sx, sy = self.start
        gx, gy = self.goal
        x, y = self.state
        if 0 <= sx < self.width and 0 <= sy < self.height:
            grid[sy][sx] = "S"
        if 0 <= gx < self.width and 0 <= gy < self.height:
            grid[gy][gx] = "G"
        if 0 <= x < self.width and 0 <= y < self.height:
            grid[y][x] = "R"
        return "\n".join(" ".join(row) for row in grid)

    def get_state(self):
        return self.state

    def is_done(self):
        done_goal = self.state == self.goal and set(self.waypoints).issubset(self.visited_waypoints)
        done_timeout = self.max_steps is not None and self.steps >= self.max_steps
        return done_goal or done_timeout

    # ----------------- helper for A2C -----------------
    def build_grid_state(self):
        """Trả về tensor 5 kênh (robot, goal, obstacles, waypoint chưa thăm, waypoint đã thăm)"""
        grid = torch.zeros(5, self.height, self.width, dtype=torch.float32)
        rx, ry = self.state
        gx, gy = self.goal
        grid[0, ry, rx] = 1.0
        grid[1, gy, gx] = 1.0
        for ox, oy in self.obstacles:
            if 0 <= ox < self.width and 0 <= oy < self.height:
                grid[2, oy, ox] = 1.0
        for wx, wy in self.waypoints:
            if (wx, wy) not in self.visited_waypoints:
                grid[3, wy, wx] = 1.0
            else:
                grid[4, wy, wx] = 1.0
        return grid
    
    def step_to(self, target):
        """
        Di chuyển trực tiếp robot tới ô target (x,y), cập nhật steps và visited_waypoints.
        Không kiểm tra action hợp lệ như RL, dùng để chạy A*.
        """
        self.state = target
        self.steps += 1
        if target in self.waypoints:
            self.visited_waypoints.add(target)
        done = (target == self.goal)
        reward = 1 if target in self.waypoints else 0
        info = {"note": "Auto move by A*"}
        return target, reward, done, info

