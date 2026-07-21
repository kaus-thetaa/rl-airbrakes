import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RocketEnv(gym.Env):
    """
    rocket airbrake control environment
    agent acts during coast phase (burnout → apogee)
    controls airbrake extension to hit target apogee
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # sim timestep and episode length
        self.dt = 0.05
        self.max_steps = 600
        self.target_apogee = 3000.0

        # generic sounding rocket params
        self.mass = 15.0            # kg at burnout
        self.area = 0.0079          # m^2, 10cm diameter body
        self.Cd_body = 0.45         # base drag coeff
        self.Cd_brake_max = 0.80    # extra drag at full extension
        self.rho = 0.9              # air density at ~1000m

        # state at motor burnout — agent starts here
        self.burnout_altitude = 800.0
        self.burnout_velocity = 220.0

        self.g = 9.81

        # obs: [altitude, velocity, target_alt, time, brake_extension]
        obs_low  = np.array([0.0,   -500.0, 0.0,    0.0, 0.0], dtype=np.float32)
        obs_high = np.array([6000.0, 500.0, 6000.0, 60.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # single continuous action — how open the brakes are
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # internal state vars
        self.altitude = 0.0
        self.velocity = 0.0
        self.time = 0.0
        self.steps = 0
        self.prev_extension = 0.0
        self.apogee_reached = False
        self.apogee_altitude = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # slight randomization so agent generalizes across launch conditions
        self.altitude = self.burnout_altitude + self.np_random.uniform(-50, 50)
        self.velocity = self.burnout_velocity + self.np_random.uniform(-20, 20)
        self.time = 0.0
        self.steps = 0
        self.prev_extension = 0.0
        self.apogee_reached = False
        self.apogee_altitude = 0.0

        return self._get_obs(), {}

    def step(self, action):
        extension = float(np.clip(action[0], 0.0, 1.0))

        # total drag = body + brakes
        Cd_total = self.Cd_body + extension * self.Cd_brake_max
        drag = 0.5 * self.rho * self.velocity**2 * Cd_total * self.area

        # drag always opposes velocity direction
        drag_force = -np.sign(self.velocity) * drag
        gravity_force = -self.mass * self.g

        acceleration = (drag_force + gravity_force) / self.mass

        # euler integration
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt
        self.time += self.dt
        self.steps += 1

        # lock in apogee the moment velocity flips negative
        if self.velocity <= 0 and not self.apogee_reached:
            self.apogee_reached = True
            self.apogee_altitude = self.altitude

        # episode ends on ground hit or timeout
        terminated = self.altitude <= 0.0
        truncated = self.steps >= self.max_steps

        reward = self._get_reward(extension, terminated or truncated)
        self.prev_extension = extension

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.altitude,
            self.velocity,
            self.target_apogee,
            self.time,
            self.prev_extension
        ], dtype=np.float32)

    def _get_reward(self, extension, done):
        reward = 0.0

        # small per-step penalty for jerky brake movements
        reward -= 2.0 * abs(extension - self.prev_extension)

        # big reward at episode end based on apogee accuracy
        if done and self.apogee_reached:
            apogee_error = abs(self.apogee_altitude - self.target_apogee)
            # 0 error → +100, 1000m error → 0
            reward += max(0, 100 - (apogee_error / 10.0))

        return reward