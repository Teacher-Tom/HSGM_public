import logging
import habitat_sim
import numpy as np
import magnum as mn

from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from utils import local_to_global

class PolarAction:

    def __init__(self, r, theta, type=None):
        self.theta = theta  # action angle (radians)
        self.r = r  # action radius
        self.type = type

    @classmethod
    def null(cls):
        return cls(0, 0, type='null')

    @classmethod
    def stop(cls):
        return cls(0, 0, type='stop')

    @classmethod
    def error(cls):
        return cls(0, 0, type='error')

    @classmethod
    def pause(cls):
        return cls(0, 0, type='pause')  # pause action, used to switch to the next goal

    @classmethod
    def default(cls):
        return cls(0, 0, type='default')

    def is_stop(self):
        return self.type == 'stop'

    def is_turn(self):
        return self.type in ['turn left', 'turn right', 'turn around']

    def __repr__(self):
        return f"PolarAction(r={self.r}, theta={self.theta}, type={self.type})"

PolarAction.stop = PolarAction(0, 0, 'stop')
PolarAction.null = PolarAction(0, 0, 'null')
    
class SimWrapper:
    """
    A wrapper for Habitat-Sim that initializes agents, sensors, and manages movement and sensor observations.
    """
    def __init__(self, cfg):
        """
        Initialize the simulator with the specified configurations.

        :param cfg: Dictionary with configurations for the simulator, agents, and sensors.
        """
        self.scene_id = cfg['scene_id']
        self.use_goal_image_agent = cfg['use_goal_image_agent']
        self.allow_slide = cfg['allow_slide']
        self.sensor_pitch = cfg['sensor_cfg']['pitch']

        self.sensor_pitch = np.deg2rad(self.sensor_pitch) if self.sensor_pitch is not None else 0.0
        self.fov = cfg['sensor_cfg']['fov']
        self.sensor_height = cfg['sensor_cfg']['height']
        self.agent_radius = cfg['agent_radius']
        self.agent_height = cfg['agent_height']
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )

        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = cfg['scene_path']
        if 'scene_config' in cfg:
            backend_cfg.scene_dataset_config_file = cfg['scene_config']
        backend_cfg.enable_physics = True

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = self.agent_radius  # agent radius
        agent_cfg.height = self.agent_height
        agent_cfg.sensor_specifications = self._create_sensor_specs()

        agents = [agent_cfg]
        if self.use_goal_image_agent:
            goal_cfg = self._create_goal_agent_config(fov=cfg['goal_image_agent_fov'])
            agents.append(goal_cfg)
        self.panorama_sensor_fov = 90.0
        self.panorama_resolution = (540, 540)
        self.panorama_view_specs = [
            ('panorama_left', np.pi / 2),
            ('panorama_front', 0.0),
            ('panorama_right', -np.pi / 2),
        ]
        self.panorama_agent_id = len(agents)
        agents.append(self._create_panorama_agent_config(fov=self.panorama_sensor_fov))

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, agents)

        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
            logging.info('Simulator initialized!')
        except Exception as e:
            logging.error(f"Error initializing simulator: {e}")
            raise RuntimeError(f"Could not initialize simulator: {e}")

    def _set_initial_agent_pos_rot(self, position, rotation):
        """
        Set agent initial position and rotation.

        :param position: agent position (numpy array or list)
        :param rotation: agent rotation (quaternion)
        """
        agent = self.sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = np.array(position)
        agent_state.rotation = rotation
        agent.set_state(agent_state)

    def _create_sensor_specs(self):
        """
        Create sensor specifications for the agent.

        :return: List of sensor specifications.
        """
        sensor_specs = []
        sensor_specs.append(self._create_rgb_sensor_spec())
        sensor_specs.append(self._create_depth_sensor_spec())
        return sensor_specs

    def _create_rgb_sensor_spec(self):
        """
        Create an RGB camera sensor specification.

        :return: RGB sensor specification.
        """
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f"color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.resolution
        rgb_sensor_spec.hfov = self.fov
        rgb_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        rgb_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return rgb_sensor_spec

    def _create_depth_sensor_spec(self):
        """
        Create a depth camera sensor specification.

        :return: Depth sensor specification.
        """
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = f"depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.resolution
        depth_sensor_spec.hfov = self.fov
        depth_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        depth_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return depth_sensor_spec

    def _create_panorama_agent_config(self, fov):
        """
        Create an auxiliary agent with a 90-degree square RGB/depth sensor.
        The main agent sensors keep their original FOV; this agent is only used
        to capture the stitched panoramic observation panels.
        """
        pano_cfg = habitat_sim.agent.AgentConfiguration()
        pano_cfg.radius = self.agent_radius
        pano_cfg.height = self.agent_height

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.panorama_resolution
        rgb_sensor_spec.hfov = fov
        rgb_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        rgb_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.panorama_resolution
        depth_sensor_spec.hfov = fov
        depth_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        depth_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])

        pano_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        return pano_cfg

    def _create_goal_agent_config(self, fov):
        """
        Create a configuration for the goal image capture agent.

        :param fov: Field of view for the goal agent.
        :return: Agent configuration for the goal agent.
        """
        goal_cfg = habitat_sim.agent.AgentConfiguration()
        goal_sensor_spec = habitat_sim.CameraSensorSpec()
        goal_sensor_spec.uuid = "goal_sensor"
        goal_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        goal_sensor_spec.resolution = self.resolution
        goal_sensor_spec.hfov = fov
        goal_sensor_spec.orientation = mn.Vector3([0, 0, 0])
        goal_sensor_spec.position = mn.Vector3([0, 0, 0])
        goal_cfg.sensor_specifications = [goal_sensor_spec]
        return goal_cfg

    @staticmethod
    def camera_intrinsic_for_hfov(width, height, hfov_deg):
        """
        Build a pinhole intrinsic matrix for a Habitat camera sensor.

        width/height are image dimensions in pixels. Habitat sensor specs store
        horizontal FOV, so fx and fy share the same focal length here.
        """
        safe_hfov = min(float(hfov_deg), 179.0)
        f = (float(width) / 2.0) / np.tan(np.deg2rad(safe_hfov / 2.0))
        return np.array([
            [f, 0, float(width) / 2.0],
            [0, f, float(height) / 2.0],
            [0, 0, 1],
        ], dtype=np.float32)

    def get_main_camera_intrinsic(self):
        height, width = self.resolution
        return self.camera_intrinsic_for_hfov(width, height, self.fov)

    def get_panorama_camera_intrinsic(self):
        height, width = self.panorama_resolution
        return self.camera_intrinsic_for_hfov(width, height, self.panorama_sensor_fov)

    def step(self, action: PolarAction):
        """
        Move the agent based on the specified action and magnitude.

        :param action: The action to perform.
        :return: Dictionary containing observations from front, left, right, and back perspectives.
        """
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()

        if action is PolarAction.null:
            new_agent_state = curr_state
        else:
            new_agent_state = habitat_sim.AgentState()
            new_agent_state.position = np.copy(curr_state.position)
            new_agent_state.rotation = curr_state.rotation
            self._rotate_yaw(new_agent_state, new_agent_state.rotation, action.theta)
            self._move_forward(new_agent_state, new_agent_state.position, new_agent_state.rotation, action.r)
            agent.set_state(new_agent_state)

        observation = self.sim.get_sensor_observations(0)
        observation['agent_state'] = agent.get_state()

        current_state = agent.get_state()

        left_state = habitat_sim.AgentState()
        left_state.position = np.copy(current_state.position)
        left_state.rotation = current_state.rotation
        self._rotate_yaw(left_state, current_state.rotation, np.pi / 2)
        agent.set_state(left_state)
        observation['left'] = self.sim.get_sensor_observations(0)
        observation['left']['agent_state'] = agent.get_state()

        right_state = habitat_sim.AgentState()
        right_state.position = np.copy(current_state.position)
        right_state.rotation = current_state.rotation
        self._rotate_yaw(right_state, current_state.rotation, -np.pi / 2)
        agent.set_state(right_state)
        observation['right'] = self.sim.get_sensor_observations(0)
        observation['right']['agent_state'] = agent.get_state()

        back_state = habitat_sim.AgentState()
        back_state.position = np.copy(current_state.position)
        back_state.rotation = current_state.rotation
        self._rotate_yaw(back_state, current_state.rotation, np.pi)
        agent.set_state(back_state)
        observation['back'] = self.sim.get_sensor_observations(0)
        observation['back']['agent_state'] = agent.get_state()

        agent.set_state(current_state)

        return observation

    def _move_forward(self, agent_state, curr_position, curr_rotation, magnitude):
        """
        Move the agent_state forward by a specified magnitude.

        :param agent_state: The state of the agent to be updated.
        :param curr_position: Current position of the agent.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Distance to move forward.
        """
        local_point = np.array([0, 0, -magnitude])
        global_point = local_to_global(curr_position, curr_rotation, local_point)
        delta = (global_point - curr_position) / 10
        new_position = np.copy(curr_position)

        for _ in range(10):
            if self.allow_slide:
                new_position = self.sim.pathfinder.try_step(new_position, new_position + delta)
            else:
                new_position = self.sim.pathfinder.try_step_no_sliding(new_position, new_position + delta)

        agent_state.position = new_position

    def _rotate_yaw(self, agent_state, curr_rotation, magnitude):
        """
        Rotate the agent_state by a specified angle.

        :param agent_state: The state of the agent to be updated.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Angle in radians to rotate counterclockwise.
        """
        theta, axis = quat_to_angle_axis(curr_rotation)
        if axis[1] < 0:  # Ensure consistent rotation direction
            theta = 2 * np.pi - theta
        new_theta = theta + magnitude
        agent_state.rotation = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))

    def get_goal_image(self, goal_position, goal_rotation):
        """
        Capture an image from the goal agent's perspective.

        :param goal_position: Position of the goal agent.
        :param goal_rotation: Rotation of the goal agent.
        :return: The captured image from the goal sensor.
        """
        assert self.use_goal_image_agent, "Goal image agent is not enabled."

        goal_agent = self.sim.get_agent(1)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        goal_agent.set_state(new_agent_state)
        return self.sim.get_sensor_observations(1)['goal_sensor']

    def set_random_valid_state(self):
        """
        Set the agent to a random valid starting position.
        """
        agent = self.sim.get_agent(0)
        for _ in range(100):
            random_pos = self.sim.pathfinder.get_random_navigable_point()
            agent_state = agent.get_state()
            agent_state.position = random_pos
            agent.set_state(agent_state)
            if self.sim.pathfinder.is_navigable(agent_state.position):
                logging.info(f"Agent set to random valid state at: {random_pos}")
                return
        logging.warning("Could not find a valid random starting point after 100 attempts.")

    def get_obs(self):
        """
        Get the current sensor observations for agent 0.
        """
        return self.sim.get_sensor_observations(0)

    def get_panorama_observations(self):
        """
        Capture three 90-degree square observations at current heading: left, front, right.
        These come from an independent panorama agent; does not affect main sensor FOV.

        :return: Dictionary containing left/front/right panorama panel observations.
        """
        main_agent = self.sim.get_agent(0)
        current_state = main_agent.get_state()
        panorama_agent = self.sim.get_agent(self.panorama_agent_id)

        observations = {}
        for view_name, yaw_offset in self.panorama_view_specs:
            pano_state = habitat_sim.AgentState()
            pano_state.position = np.copy(current_state.position)
            pano_state.rotation = current_state.rotation
            self._rotate_yaw(pano_state, current_state.rotation, yaw_offset)
            panorama_agent.set_state(pano_state)

            pano_obs = self.sim.get_sensor_observations(self.panorama_agent_id)
            pano_obs['agent_state'] = panorama_agent.get_state()
            observations[view_name] = pano_obs

        return observations

    def move_forward(self, distance: float):
        """
        Move the agent forward by a specified distance.
        """
        self.step(PolarAction(r=distance, theta=0))

    def turn_angle(self, angle_deg: float):
        """
        Turn the agent by a specified angle in degrees.
        """
        self.step(PolarAction(r=0, theta=np.deg2rad(angle_deg)))

    def reset(self):
        """
        Reset the simulator.
        """
        self.sim.reset()

    def close(self):
        try:
            self.sim.close()
        except:
            pass

    def set_initial_state(self, position, rotation=None):
        """
        Set agent initial state.

        :param position: agent position (numpy array or list)
        :param rotation: agent rotation (quaternion). If None, uses the default rotation.
        """
        agent = self.sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = np.array(position)
        if rotation is not None:
            agent_state.rotation = rotation
        agent.set_state(agent_state)
        pass

    def get_path(self, path):
        """
        Find a path to the specified target position.

        :param path: Target position for pathfinding.
        :return: The path found by the pathfinder.
        """
        if self.sim.pathfinder.find_path(path):
            return path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            return 1000
