"""
SkillMimic Dual Humanoid Environment

This module implements a dual-humanoid environment for cooperative pass-and-catch tasks.
Actor structure per environment:
  - Actor 0: Humanoid A (passer)
  - Actor 1: Humanoid B (catcher)  
  - Actor 2: Ball (target)
  - Actor 3+: Projectiles (optional)

Author: Cursor Agent
Created for: Pass-and-Catch cooperative task extension
"""

from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import glob, os, random
from datetime import datetime

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject
from env.tasks.humanoid_task import compute_humanoid_observations

# Size of the other humanoid's relative state observation
# Includes: relative_pos(3) + relative_rot(6, tan-norm) + relative_vel(3) + relative_ang_vel(3) = 15
OTHER_HUMANOID_OBS_SIZE = 15

# Default reward weights for cooperative task
DEFAULT_COOP_REWARD_WEIGHTS = {
    "alive": 0.1,
    "ball_to_hand": 2.0,
    "pass_direction": 5.0,      # Increased from 1.0 for better pass targeting
    "catch_success": 10.0,
    "ball_height": 0.5,
    "standing": 2.0,            # NEW: CoM height reward
    "upright": 1.5,             # NEW: Body verticality reward
    "ground_contact_penalty": -5.0,  # NEW: Penalty for non-foot ground contact
}

# Minimum standing height for rewards (meters)
MIN_STANDING_HEIGHT = 0.8

# Body indices that should NOT touch ground (will be set from asset)
# Typical non-foot bodies: head, torso, hands, knees
NON_FOOT_BODY_NAMES = ["Head", "Pelvis", "Spine", "Spine1", "Spine2", 
                        "L_Hand", "R_Hand", "L_Forearm", "R_Forearm",
                        "L_Knee", "R_Knee"]


class SkillMimicDualHumanoid(HumanoidWholeBodyWithObject):
    """
    Dual Humanoid environment for cooperative pass-and-catch tasks.
    
    This class extends HumanoidWholeBodyWithObject to support two humanoid actors
    in each environment. The observation and reward computations are initially
    set to dummy implementations to ensure physics stability first.
    """
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Configuration for dual humanoid setup
        self._humanoid_b_spacing = cfg["env"].get("humanoidBSpacing", 2.0)  # Offset for humanoid B
        
        state_init = str(cfg["env"]["stateInit"])
        if state_init.lower() == "random":
            self._state_init = -1
            print("Random Reference State Init (RRSI)")
        else:
            self._state_init = int(state_init)
            print(f"Deterministic Reference State Init from {self._state_init}")

        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
        self.reward_weights_default = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.save_images_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.init_vel = cfg['env']['initVel']
        self.isTest = cfg['args'].test

        self.condition_size = 64

        # Call parent constructor (this will call _create_envs)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6

        # Build tensors for humanoid B (humanoid A tensors are built in parent)
        self._build_humanoid_b_tensors()
        
        # Build hand body IDs for ball contact detection
        self._build_hand_body_ids()
        
        # Load cooperative reward weights
        self._load_coop_reward_weights()

        # Motion data loading (simplified for dual humanoid)
        self._load_motion(self.motion_file)

        # Initialize observation buffers
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)

        # Initialize state buffers for humanoid B
        self.init_root_pos_b = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_root_rot_b = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self.init_root_pos_vel_b = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_root_rot_vel_b = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_dof_pos_b = torch.zeros([self.num_envs, self.num_dof], device=self.device, dtype=torch.float)
        self.init_dof_pos_vel_b = torch.zeros([self.num_envs, self.num_dof], device=self.device, dtype=torch.float)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        print(f"[SkillMimicDualHumanoid] Initialized with {self.num_envs} environments")
        print(f"[SkillMimicDualHumanoid] Actors per env: {self.get_num_actors_per_env()}")
        print(f"[SkillMimicDualHumanoid] Humanoid B spacing: {self._humanoid_b_spacing}m")

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Override to create dual humanoid environments.
        
        Actor creation order per environment:
        1. Humanoid A (at origin)
        2. Humanoid B (offset by _humanoid_b_spacing on X axis)
        3. Ball (target)
        4. Projectiles (optional)
        """
        self._target_handles = []
        self._humanoid_b_handles = []  # Track handles for humanoid B
        
        self._load_target_asset()
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()
        
        # Load humanoid asset
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # Force sensors
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []  # Humanoid A handles
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # Aggregate size: 2 humanoids + ball + optional projectiles
        max_agg_bodies = self.num_humanoid_bodies * 2 + 2
        max_agg_shapes = self.num_humanoid_shapes * 2 + 2
        
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env_dual(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # DOF limits
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if self._pd_control:
            self._build_pd_action_offset_scale()

        return

    def _build_env_dual(self, env_id, env_ptr, humanoid_asset):
        """
        Build a single environment with two humanoids, ball, and optional projectiles.
        
        Creation order:
        1. Humanoid A (actor index 0)
        2. Humanoid B (actor index 1)
        3. Ball (actor index 2)
        4. Projectiles (actor index 3+)
        """
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89

        # === Humanoid A (at origin) ===
        start_pose_a = gymapi.Transform()
        start_pose_a.p = gymapi.Vec3(0.0, 0.0, char_h)
        start_pose_a.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_a_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose_a, 
                                                   "humanoid_a", col_group, col_filter, segmentation_id)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_a_handle)

        # Set color for humanoid A (green)
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_a_handle, j, gymapi.MESH_VISUAL, 
                                          gymapi.Vec3(0.54, 0.85, 0.2))

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_a_handle, dof_prop)

        self.humanoid_handles.append(humanoid_a_handle)

        # === Humanoid B (offset position) ===
        start_pose_b = gymapi.Transform()
        start_pose_b.p = gymapi.Vec3(self._humanoid_b_spacing, 0.0, char_h)
        start_pose_b.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)  # Facing opposite direction (180 deg rotation around Z)

        humanoid_b_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose_b, 
                                                   "humanoid_b", col_group, col_filter, segmentation_id)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_b_handle)

        # Set color for humanoid B (blue)
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_b_handle, j, gymapi.MESH_VISUAL, 
                                          gymapi.Vec3(0.2, 0.54, 0.85))

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_b_handle, dof_prop)

        self._humanoid_b_handles.append(humanoid_b_handle)

        # === Ball (target) ===
        self._build_target(env_id, env_ptr)

        # === Projectiles (optional) ===
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

        return

    def _build_humanoid_b_tensors(self):
        """
        Build tensor views for humanoid B after simulation is created.
        
        Actor indices in the flattened root state tensor:
        - Humanoid A: num_actors * env_id + 0
        - Humanoid B: num_actors * env_id + 1
        - Ball: num_actors * env_id + 2
        """
        num_actors = self.get_num_actors_per_env()
        
        # Actor IDs for humanoid B (actor index 1 in each env)
        self._humanoid_b_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1
        
        # Root states view for humanoid B
        self._humanoid_b_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        # Initial root states for humanoid B
        self._initial_humanoid_b_root_states = self._humanoid_b_root_states.clone()
        self._initial_humanoid_b_root_states[:, 7:13] = 0  # Zero velocities
        
        # DOF states for humanoid B
        # Each env has: humanoid_a_dofs + humanoid_b_dofs
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos_b = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., self.num_dof:self.num_dof*2, 0]
        self._dof_vel_b = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., self.num_dof:self.num_dof*2, 1]
        
        # Rigid body states for humanoid B
        # Bodies per env: humanoid_a_bodies + humanoid_b_bodies + ball_body (+ projectile_bodies)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        
        # Humanoid B rigid bodies start at index num_bodies
        self._rigid_body_pos_b = rigid_body_state_reshaped[..., self.num_bodies:self.num_bodies*2, 0:3]
        self._rigid_body_rot_b = rigid_body_state_reshaped[..., self.num_bodies:self.num_bodies*2, 3:7]
        self._rigid_body_vel_b = rigid_body_state_reshaped[..., self.num_bodies:self.num_bodies*2, 7:10]
        self._rigid_body_ang_vel_b = rigid_body_state_reshaped[..., self.num_bodies:self.num_bodies*2, 10:13]
        
        # Contact forces for humanoid B
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces_b = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies:self.num_bodies*2, :]

        print(f"[SkillMimicDualHumanoid] Built humanoid B tensors:")
        print(f"  - num_actors per env: {num_actors}")
        print(f"  - bodies per env: {bodies_per_env}")
        print(f"  - DOFs per humanoid: {self.num_dof}")
        
        return

    def _compute_humanoid_b_obs(self, env_ids=None):
        """
        Compute full-body observations for humanoid B in the same format and
        dimension as the single-humanoid `humanoid_obs` used by the LLC.
        
        This mirrors `HumanoidWholeBodyWithObject._compute_humanoid_obs`, but
        uses humanoid B's rigid body states and contact forces instead of A's.
        """
        if env_ids is None:
            body_pos = self._rigid_body_pos_b
            body_rot = self._rigid_body_rot_b
            body_vel = self._rigid_body_vel_b
            body_ang_vel = self._rigid_body_ang_vel_b
            contact_forces = self._contact_forces_b
        else:
            body_pos = self._rigid_body_pos_b[env_ids]
            body_rot = self._rigid_body_rot_b[env_ids]
            body_vel = self._rigid_body_vel_b[env_ids]
            body_ang_vel = self._rigid_body_ang_vel_b[env_ids]
            contact_forces = self._contact_forces_b[env_ids]

        obs = compute_humanoid_observations(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            self._local_root_obs,
            self._root_height_obs,
            contact_forces,
            self._contact_body_ids,
        )

        return obs

    def _build_hand_body_ids(self):
        """
        Build tensor of hand body IDs for ball catching detection.
        
        Hand bodies are used to:
        1. Compute distance from ball to hands
        2. Detect contact between ball and hands
        """
        env_ptr = self.envs[0]
        humanoid_handle = self.humanoid_handles[0]
        
        # Get hand body names from config
        hand_body_names = self.cfg["env"].get("handBodies", [
            "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3",
            "R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3", "R_Thumb3"
        ])
        
        # Find body IDs for hand bodies
        hand_body_ids = []
        for body_name in hand_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, humanoid_handle, body_name)
            if body_id != -1:
                hand_body_ids.append(body_id)
            else:
                print(f"[Warning] Hand body '{body_name}' not found")
        
        self._hand_body_ids = to_torch(hand_body_ids, device=self.device, dtype=torch.long)
        
        # Separate left and right hand IDs
        left_hand_names = ["L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3"]
        right_hand_names = ["R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3", "R_Thumb3"]
        
        left_ids = []
        right_ids = []
        for body_name in left_hand_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, humanoid_handle, body_name)
            if body_id != -1:
                left_ids.append(body_id)
        for body_name in right_hand_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, humanoid_handle, body_name)
            if body_id != -1:
                right_ids.append(body_id)
        
        self._left_hand_body_ids = to_torch(left_ids, device=self.device, dtype=torch.long)
        self._right_hand_body_ids = to_torch(right_ids, device=self.device, dtype=torch.long)
        
        print(f"[SkillMimicDualHumanoid] Hand body IDs: {len(self._hand_body_ids)}")
        print(f"  - Left hand bodies: {len(self._left_hand_body_ids)}")
        print(f"  - Right hand bodies: {len(self._right_hand_body_ids)}")
        
        return

    def _load_coop_reward_weights(self):
        """Load cooperative reward weights from config (separate from imitation rewards)."""
        # Use coopRewardWeights for cooperative task (separate from rewardWeights used by MotionDataHandler)
        coop_weights = self.cfg["env"].get("coopRewardWeights", {})
        
        self._reward_w_alive = coop_weights.get("alive", DEFAULT_COOP_REWARD_WEIGHTS["alive"])
        self._reward_w_ball_to_hand = coop_weights.get("ball_to_hand", DEFAULT_COOP_REWARD_WEIGHTS["ball_to_hand"])
        self._reward_w_pass_direction = coop_weights.get("pass_direction", DEFAULT_COOP_REWARD_WEIGHTS["pass_direction"])
        self._reward_w_catch_success = coop_weights.get("catch_success", DEFAULT_COOP_REWARD_WEIGHTS["catch_success"])
        self._reward_w_ball_height = coop_weights.get("ball_height", DEFAULT_COOP_REWARD_WEIGHTS["ball_height"])
        self._reward_w_standing = coop_weights.get("standing", DEFAULT_COOP_REWARD_WEIGHTS["standing"])
        self._reward_w_upright = coop_weights.get("upright", DEFAULT_COOP_REWARD_WEIGHTS["upright"])
        self._reward_w_ground_contact_penalty = coop_weights.get("ground_contact_penalty", DEFAULT_COOP_REWARD_WEIGHTS["ground_contact_penalty"])
        
        # Build non-foot body indices for ground contact detection
        self._build_non_foot_body_ids()
        
        print(f"[SkillMimicDualHumanoid] Cooperative reward weights:")
        print(f"  - alive: {self._reward_w_alive}")
        print(f"  - ball_to_hand: {self._reward_w_ball_to_hand}")
        print(f"  - pass_direction: {self._reward_w_pass_direction}")
        print(f"  - catch_success: {self._reward_w_catch_success}")
        print(f"  - ball_height: {self._reward_w_ball_height}")
        print(f"  - standing: {self._reward_w_standing}")
        print(f"  - upright: {self._reward_w_upright}")
        print(f"  - ground_contact_penalty: {self._reward_w_ground_contact_penalty}")
    
    def _build_non_foot_body_ids(self):
        """Build list of body indices that should not touch ground (non-foot bodies)."""
        non_foot_ids = []
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        
        for body_name in NON_FOOT_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            if body_id != -1:
                non_foot_ids.append(body_id)
        
        self._non_foot_body_ids = torch.tensor(non_foot_ids, device=self.device, dtype=torch.long)
        print(f"[SkillMimicDualHumanoid] Non-foot body IDs for ground contact detection: {self._non_foot_body_ids.tolist()}")

    def _build_target_tensors(self):
        """
        Override to account for dual humanoid setup.
        Ball is now at actor index 2 (after both humanoids).
        """
        num_actors = self.get_num_actors_per_env()
        
        # Ball is at actor index 2
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        
        # Ball actor IDs
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2
        
        # Ball contact forces
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # Ball rigid body is at index: num_bodies * 2 (after both humanoids)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies * 2, :]

        # Initial object states
        self.init_obj_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.init_obj_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        return

    def _load_motion(self, motion_file):
        """Load motion data (simplified for dual humanoid testing)."""
        self.skill_name = os.path.basename(motion_file)
        self.max_episode_length = 60
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length = self.cfg["env"]["episodeLength"]

        self._motion_data = MotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                              self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset)
        
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], 
                                          device=self.device, dtype=torch.float)
        return

    def get_action_size(self):
        """
        Override action size for dual humanoid.
        
        Action space is doubled to control both humanoids independently:
        - First half: actions for Humanoid A (156 dims)
        - Second half: actions for Humanoid B (156 dims)
        """
        single_humanoid_actions = self._num_actions  # 156 for mocap_humanoid
        return single_humanoid_actions * 2  # 312 total

    def get_obs_size(self):
        """
        Override observation size for dual humanoid.
        
        Observation structure for each humanoid:
        - Self humanoid state (from parent class)
        - Object (ball) state (from parent class)  
        - Other humanoid relative state (NEW: 15 dims)
        - Condition embedding (64 dims)
        
        For now, we output a combined observation.
        Both humanoids share the same observation (can be separated for MAPPO later).
        """
        obs_size = super().get_obs_size()
        obs_size += OTHER_HUMANOID_OBS_SIZE  # Add other humanoid's relative state
        obs_size += self.condition_size
        return obs_size

    def _reset_actors(self, env_ids):
        """Reset both humanoids and the ball."""
        if self._state_init == -1:
            self._reset_random_ref_state_init(env_ids)
        elif self._state_init >= 2:
            self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert False, f"Unsupported state initialization from: {self._state_init}"

        # Reset humanoid A
        self._reset_humanoid_a(env_ids)
        
        # Reset humanoid B
        self._reset_humanoid_b(env_ids)
        
        # Reset ball
        self._reset_target(env_ids)
        
        return

    def _reset_humanoid_a(self, env_ids):
        """Reset humanoid A to initial state."""
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return

    def _reset_humanoid_b(self, env_ids):
        """
        Reset humanoid B to the LEFT of humanoid A, facing A (side-by-side, facing each other).
        
        This ensures that:
        1. B is always to A's LEFT side (perpendicular to A's forward direction)
        2. B faces A (90 degrees clockwise from A's heading, so B faces right toward A)
        3. B maintains the same upright posture as A (preserves pitch/roll from motion data)
        
        Coordinate system (Isaac Gym):
        - X axis: forward
        - Y axis: left
        - Z axis: up
        
        If A's heading is theta (from +X axis, counterclockwise):
        - A's forward direction: (cos(theta), sin(theta), 0)
        - A's left direction: (-sin(theta), cos(theta), 0)
        - B's position = A's position + A's left direction × spacing
        - B's heading = A's heading + 90° (B faces right, toward A)
        """
        num_reset = env_ids.shape[0]
        
        # Get A's full rotation and heading from motion data
        root_rot_a = self.init_root_rot[env_ids]
        heading_a = torch_utils.calc_heading(root_rot_a)  # [N] heading angle in radians
        
        # B's position = A's position + A's LEFT direction × spacing
        # Left direction: (-sin(heading), cos(heading), 0) - perpendicular to forward
        offset_x = -torch.sin(heading_a) * self._humanoid_b_spacing
        offset_y = -torch.cos(heading_a) * self._humanoid_b_spacing
        
        self._humanoid_b_root_states[env_ids, 0] = self.init_root_pos[env_ids, 0] + offset_x
        self._humanoid_b_root_states[env_ids, 1] = self.init_root_pos[env_ids, 1] + offset_y
        self._humanoid_b_root_states[env_ids, 2] = self.init_root_pos[env_ids, 2]
        
        # B's rotation: Keep A's pitch/roll (upright posture), but rotate heading by +90°
        # This makes B face RIGHT (toward A, who is on B's right side)
        # Method: Decompose A's rotation into heading + pitch/roll, then recompose with new heading
        # Extract A's heading rotation
        heading_quat_a = torch_utils.calc_heading_quat(root_rot_a)
        heading_quat_a_inv = torch_utils.quat_conjugate(heading_quat_a)
        
        # Remove heading from A's rotation to get pitch/roll component
        pitch_roll_quat = quat_mul(heading_quat_a_inv, root_rot_a)
        
        # Create B's heading: B faces RIGHT (toward A, who is on B's right side)
        # If A's heading is theta, B's heading should be theta - 90° (or theta + 270°)
        # This makes B face right, which is toward A
        heading_b = heading_a + np.pi  # -90 degrees (face right toward A)
        axis = torch.zeros((num_reset, 3), device=self.device)
        axis[:, 2] = 1.0  # Z axis
        heading_quat_b = quat_from_angle_axis(heading_b, axis)
        
        # Recompose: B's rotation = B's heading * A's pitch/roll
        rot_b = quat_mul(heading_quat_b, pitch_roll_quat)
        self._humanoid_b_root_states[env_ids, 3:7] = rot_b
        
        # DEBUG: Print rotation info
        if num_reset > 0:
            print(f"[DEBUG _reset_humanoid_b] num_reset: {num_reset}")
            print(f"[DEBUG] A init_root_pos[0]: {self.init_root_pos[env_ids[0]]}")
            print(f"[DEBUG] A heading_a[0]: {heading_a[0]:.4f} ({heading_a[0]*180/np.pi:.2f}°)")
            print(f"[DEBUG] B position offset: ({offset_x[0]:.3f}, {offset_y[0]:.3f}, 0)")
            print(f"[DEBUG] B heading_b[0]: {heading_b[0]:.4f} ({heading_b[0]*180/np.pi:.2f}°)")
            print(f"[DEBUG] B rot_b[0]: {rot_b[0]}")
        
        # Zero velocities
        self._humanoid_b_root_states[env_ids, 7:13] = 0.0
        
        # Same DOF poses as humanoid A (can be customized later)
        self._dof_pos_b[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel_b[env_ids] = 0.0
        
        return

    def _reset_random_ref_state_init(self, env_ids):
        """Random reference state initialization."""
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)

        skill_label = self._motion_data.motion_class[motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), 
                                                       num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids], \
        self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], \
        self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return

    def _reset_deterministic_ref_state_init(self, env_ids):
        """Deterministic reference state initialization."""
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        skill_label = self._motion_data.motion_class[motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), 
                                                       num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids], \
        self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], \
        self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return

    def _reset_env_tensors(self, env_ids):
        """
        Reset environment tensors for both humanoids and ball.
        Override to include humanoid B in the reset.
        """
        # Combine actor IDs for all actors that need to be reset
        env_ids_int32_a = self._humanoid_actor_ids[env_ids]
        env_ids_int32_b = self._humanoid_b_actor_ids[env_ids]
        env_ids_int32_ball = self._tar_actor_ids[env_ids]
        
        # Reset all root states at once
        all_actor_ids = torch.cat([env_ids_int32_a, env_ids_int32_b, env_ids_int32_ball])
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(all_actor_ids), 
                                                     len(all_actor_ids))
        
        # Reset DOF states for both humanoids
        # Note: set_dof_state_tensor_indexed uses humanoid actor IDs
        all_humanoid_ids = torch.cat([env_ids_int32_a, env_ids_int32_b])
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(all_humanoid_ids), 
                                              len(all_humanoid_ids))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _compute_other_humanoid_obs(self, env_ids=None):
        """
        Compute the relative state of the other humanoid from the perspective of self.
        
        For Humanoid A: computes Humanoid B's relative state
        For Humanoid B: computes Humanoid A's relative state
        
        Returns two tensors: (other_obs_for_A, other_obs_for_B)
        Each tensor contains: relative_pos(3) + relative_rot(6) + relative_vel(3) + relative_ang_vel(3) = 15 dims
        """
        if env_ids is None:
            root_pos_a = self._humanoid_root_states[:, 0:3]
            root_rot_a = self._humanoid_root_states[:, 3:7]
            root_vel_a = self._humanoid_root_states[:, 7:10]
            root_ang_vel_a = self._humanoid_root_states[:, 10:13]
            
            root_pos_b = self._humanoid_b_root_states[:, 0:3]
            root_rot_b = self._humanoid_b_root_states[:, 3:7]
            root_vel_b = self._humanoid_b_root_states[:, 7:10]
            root_ang_vel_b = self._humanoid_b_root_states[:, 10:13]
        else:
            root_pos_a = self._humanoid_root_states[env_ids, 0:3]
            root_rot_a = self._humanoid_root_states[env_ids, 3:7]
            root_vel_a = self._humanoid_root_states[env_ids, 7:10]
            root_ang_vel_a = self._humanoid_root_states[env_ids, 10:13]
            
            root_pos_b = self._humanoid_b_root_states[env_ids, 0:3]
            root_rot_b = self._humanoid_b_root_states[env_ids, 3:7]
            root_vel_b = self._humanoid_b_root_states[env_ids, 7:10]
            root_ang_vel_b = self._humanoid_b_root_states[env_ids, 10:13]
        
        # Compute relative observations for Humanoid A (observing B)
        other_obs_for_a = compute_other_humanoid_obs(
            root_pos_a, root_rot_a, root_vel_a, root_ang_vel_a,
            root_pos_b, root_rot_b, root_vel_b, root_ang_vel_b
        )
        
        # Compute relative observations for Humanoid B (observing A)
        other_obs_for_b = compute_other_humanoid_obs(
            root_pos_b, root_rot_b, root_vel_b, root_ang_vel_b,
            root_pos_a, root_rot_a, root_vel_a, root_ang_vel_a
        )
        
        return other_obs_for_a, other_obs_for_b

    def _compute_observations(self, env_ids=None):
        """
        Compute multi-agent observations for both humanoids.
        
        Each humanoid observes:
        - Its own body state (position, rotation, velocity, contacts)
        - The ball (object) state relative to itself
        - The other humanoid's root state relative to itself
        - Condition embedding
        
        Currently outputs observation for Humanoid A only (for single-policy training).
        TODO: For MAPPO, output separate obs for each agent.
        """
        # Compute Humanoid A's self observations
        humanoid_a_obs = self._compute_humanoid_obs(env_ids)
        
        # Compute object (ball) observations relative to Humanoid A
        obj_obs = self._compute_obj_obs(env_ids)
        
        # Compute other humanoid's relative state
        other_obs_for_a, other_obs_for_b = self._compute_other_humanoid_obs(env_ids)
        
        # Combine observations for Humanoid A
        # Structure: [self_state, obj_state, other_humanoid_state, condition]
        obs = torch.cat([humanoid_a_obs, obj_obs, other_obs_for_a], dim=-1)
        
        # Add condition embedding
        if env_ids is None:
            textemb_batch = self.hoi_data_label_batch
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[:] = obs
            env_ids = torch.arange(self.num_envs, device=self.device)
            ts = self.progress_buf.clone()
            self._curr_ref_obs = self.hoi_data_batch[env_ids, ts].clone()
        else:
            textemb_batch = self.hoi_data_label_batch[env_ids]
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[env_ids] = obs
            ts = self.progress_buf[env_ids].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()

        return

    def _get_hand_positions(self, humanoid='a'):
        """
        Get the average position of hand bodies for a humanoid.
        
        Args:
            humanoid: 'a' for humanoid A, 'b' for humanoid B
            
        Returns:
            Tensor [num_envs, 3]: Average hand position
        """
        if humanoid == 'a':
            # Get hand body positions for humanoid A
            hand_pos = self._rigid_body_pos[:, self._hand_body_ids, :]  # [N, num_hands, 3]
        else:
            # Get hand body positions for humanoid B
            hand_pos = self._rigid_body_pos_b[:, self._hand_body_ids, :]  # [N, num_hands, 3]
        
        # Return average position of all hand bodies
        avg_hand_pos = hand_pos.mean(dim=1)  # [N, 3]
        return avg_hand_pos

    def _get_closest_hand_distance(self, ball_pos, humanoid='a'):
        """
        Get the minimum distance from ball to any hand body.
        
        Args:
            ball_pos: Ball position [N, 3]
            humanoid: 'a' for humanoid A, 'b' for humanoid B
            
        Returns:
            Tensor [N]: Minimum distance from ball to any hand
        """
        if humanoid == 'a':
            hand_pos = self._rigid_body_pos[:, self._hand_body_ids, :]  # [N, num_hands, 3]
        else:
            hand_pos = self._rigid_body_pos_b[:, self._hand_body_ids, :]  # [N, num_hands, 3]
        
        # Compute distance from ball to each hand
        ball_pos_expanded = ball_pos.unsqueeze(1)  # [N, 1, 3]
        distances = torch.norm(hand_pos - ball_pos_expanded, dim=-1)  # [N, num_hands]
        
        # Return minimum distance
        min_distance, _ = distances.min(dim=1)  # [N]
        return min_distance

    def _compute_reward(self, actions):
        """
        Compute cooperative pass-and-catch reward.
        
        Reward components:
        1. alive: Basic survival reward for both humanoids standing
        2. ball_to_hand: Reward for catcher's hands approaching the ball
        3. pass_direction: Reward for ball velocity pointing toward catcher (enhanced)
        4. catch_success: Bonus when ball contacts catcher's hand (conditional on standing)
        5. ball_height: Reward for ball at catchable height
        6. standing: NEW - CoM height reward for maintaining standing posture
        7. upright: NEW - Body verticality reward
        8. ground_contact_penalty: NEW - Penalty for non-foot body parts touching ground
        """
        # Get ball state
        ball_pos = self._target_states[:, 0:3]
        ball_vel = self._target_states[:, 7:10]
        
        # Get humanoid positions and rotations
        root_pos_a = self._humanoid_root_states[:, 0:3]
        root_rot_a = self._humanoid_root_states[:, 3:7]
        root_pos_b = self._humanoid_b_root_states[:, 0:3]
        root_rot_b = self._humanoid_b_root_states[:, 3:7]
        
        # Get humanoid heights for alive check
        height_a = self._rigid_body_pos[:, 0, 2]  # Root body Z position
        height_b = self._rigid_body_pos_b[:, 0, 2]
        
        # Get ball contact forces for catch detection
        ball_contact_force = self._tar_contact_forces
        
        # Get hand distances to ball
        dist_ball_to_hand_a = self._get_closest_hand_distance(ball_pos, 'a')
        dist_ball_to_hand_b = self._get_closest_hand_distance(ball_pos, 'b')
        
        # Get contact forces for non-foot body ground contact detection
        contact_forces_a = self._contact_forces
        contact_forces_b = self._contact_forces_b
        
        # Compute reward using JIT function
        self.rew_buf[:] = compute_coop_reward(
            ball_pos, ball_vel,
            root_pos_a, root_pos_b,
            root_rot_a, root_rot_b,
            height_a, height_b,
            dist_ball_to_hand_a, dist_ball_to_hand_b,
            ball_contact_force,
            contact_forces_a, contact_forces_b,
            self._non_foot_body_ids,
            self._reward_w_alive,
            self._reward_w_ball_to_hand,
            self._reward_w_pass_direction,
            self._reward_w_catch_success,
            self._reward_w_ball_height,
            self._reward_w_standing,
            self._reward_w_upright,
            self._reward_w_ground_contact_penalty,
            self._termination_heights
        )
        return

    def _compute_reset(self):
        """
        Compute reset conditions for dual humanoid environment.
        
        Termination conditions:
        1. Either humanoid falls below termination height
        2. NEW: Non-foot body parts touch the ground (head, torso, hands, etc.)
        """
        self.reset_buf[:], self._terminate_buf[:] = compute_dual_humanoid_reset(
            self.reset_buf, 
            self.progress_buf,
            self._rigid_body_pos, 
            self._rigid_body_pos_b,
            self._contact_forces,
            self._contact_forces_b,
            self._non_foot_body_ids,
            self.max_episode_length,
            self._enable_early_termination, 
            self._termination_heights
        )
        return

    def pre_physics_step(self, actions):
        """
        Apply actions to both humanoids with independent control.
        
        Action tensor structure: [batch, 312]
        - actions[:, :156] -> Humanoid A
        - actions[:, 156:] -> Humanoid B
        """
        self.actions = actions.to(self.device).clone()
        
        # Split actions for each humanoid
        single_action_size = self._num_actions  # 156
        actions_a = self.actions[:, :single_action_size]
        actions_b = self.actions[:, single_action_size:]
        
        if self._pd_control:
            # Convert actions to PD targets for each humanoid
            pd_tar_a = self._action_to_pd_targets(actions_a)
            pd_tar_b = self._action_to_pd_targets(actions_b)
            
            # Combine targets for both humanoids
            pd_tar = torch.cat([pd_tar_a, pd_tar_b], dim=-1)
            
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            # Force control mode
            forces_a = actions_a * self.motor_efforts.unsqueeze(0) * self.power_scale
            forces_b = actions_b * self.motor_efforts.unsqueeze(0) * self.power_scale
            forces = torch.cat([forces_a, forces_b], dim=-1)
            
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        """Post-physics step processing."""
        self.progress_buf += 1

        self._refresh_sim_tensors()

        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False, t=0):
        """Render the environment."""
        if self.viewer:
            self._update_camera()
            self._draw_task()
            
            # Save images if enabled
            if self.save_images:
                env_ids = 0
                frame_id = t if self.play_dataset else self.progress_buf[env_ids]
                rgb_filename = "skillmimic/data/images/" + self.save_images_timestamp + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("skillmimic/data/images/" + self.save_images_timestamp, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)

        super().render(sync_frame_time)
        return

    def _draw_task(self):
        """Draw task-specific visualizations."""
        return

    def get_num_amp_obs(self):
        """Return the size of AMP observations (required by VecTaskPythonWrapper)."""
        return self.ref_hoi_obs_size


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_other_humanoid_obs(self_pos, self_rot, self_vel, self_ang_vel,
                                other_pos, other_rot, other_vel, other_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """
    Compute the relative state of another humanoid from the perspective of self.
    
    Args:
        self_pos: Self root position [N, 3]
        self_rot: Self root rotation (quaternion) [N, 4]
        self_vel: Self root velocity [N, 3]
        self_ang_vel: Self root angular velocity [N, 3]
        other_pos: Other root position [N, 3]
        other_rot: Other root rotation (quaternion) [N, 4]
        other_vel: Other root velocity [N, 3]
        other_ang_vel: Other root angular velocity [N, 3]
        
    Returns:
        Tensor [N, 15]: relative_pos(3) + relative_rot(6) + relative_vel(3) + relative_ang_vel(3)
    """
    # Get heading rotation (rotation around Z axis only) for local coordinate frame
    heading_rot_inv = torch_utils.calc_heading_quat_inv(self_rot)
    
    # Relative position in self's local frame
    rel_pos = other_pos - self_pos
    local_rel_pos = quat_rotate(heading_rot_inv, rel_pos)
    
    # Relative rotation in self's local frame (as tan-norm representation, 6 dims)
    rel_rot = quat_mul(heading_rot_inv, other_rot)
    local_rel_rot = torch_utils.quat_to_tan_norm(rel_rot)
    
    # Relative velocity in self's local frame
    rel_vel = other_vel - self_vel
    local_rel_vel = quat_rotate(heading_rot_inv, rel_vel)
    
    # Relative angular velocity in self's local frame
    rel_ang_vel = other_ang_vel - self_ang_vel
    local_rel_ang_vel = quat_rotate(heading_rot_inv, rel_ang_vel)
    
    # Concatenate all relative observations
    obs = torch.cat([local_rel_pos, local_rel_rot, local_rel_vel, local_rel_ang_vel], dim=-1)
    
    return obs


@torch.jit.script
def compute_dual_humanoid_reset(reset_buf, progress_buf, rigid_body_pos_a, rigid_body_pos_b,
                                 contact_forces_a, contact_forces_b, non_foot_body_ids,
                                 max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor) -> Tuple[Tensor, Tensor]
    """
    Compute reset conditions for dual humanoid environment.
    
    Termination conditions:
    1. Either humanoid falls below termination height
    2. Non-foot body parts touch the ground (significant contact force)
    
    Args:
        reset_buf: Current reset buffer
        progress_buf: Episode progress counter
        rigid_body_pos_a/b: Body positions for humanoid A/B [N, num_bodies, 3]
        contact_forces_a/b: Contact forces for humanoid A/B [N, num_bodies, 3]
        non_foot_body_ids: Indices of body parts that should not touch ground
        max_episode_length: Maximum episode length
        enable_early_termination: Whether to enable early termination
        termination_heights: Height threshold for falling
    """
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        # =========== 1. Height Check ===========
        # Check humanoid A height (root body)
        body_height_a = rigid_body_pos_a[:, 0, 2]
        body_fall_a = body_height_a < termination_heights
        
        # Check humanoid B height (root body)
        body_height_b = rigid_body_pos_b[:, 0, 2]
        body_fall_b = body_height_b < termination_heights
        
        # =========== 2. Non-Foot Ground Contact Check ===========
        # Check if any non-foot body part has significant ground contact
        # Contact force threshold: 50N (indicates significant ground contact)
        contact_threshold = 50.0
        
        # Get contact forces for non-foot bodies
        non_foot_contact_a = contact_forces_a[:, non_foot_body_ids, :]  # [N, num_non_foot, 3]
        non_foot_contact_b = contact_forces_b[:, non_foot_body_ids, :]
        
        # Check for significant contact (force magnitude > threshold)
        contact_magnitude_a = torch.norm(non_foot_contact_a, dim=-1)  # [N, num_non_foot]
        contact_magnitude_b = torch.norm(non_foot_contact_b, dim=-1)
        
        # Any non-foot body with significant contact triggers termination
        ground_contact_a = torch.any(contact_magnitude_a > contact_threshold, dim=-1)
        ground_contact_b = torch.any(contact_magnitude_b > contact_threshold, dim=-1)
        
        # =========== Combine Termination Conditions ===========
        # Either falling OR non-foot ground contact triggers termination
        has_failed = (body_fall_a | body_fall_b | ground_contact_a | ground_contact_b).clone()
        has_failed *= (progress_buf > 1)  # Don't terminate on first step
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_coop_reward(ball_pos, ball_vel,
                        root_pos_a, root_pos_b,
                        root_rot_a, root_rot_b,
                        height_a, height_b,
                        dist_ball_to_hand_a, dist_ball_to_hand_b,
                        ball_contact_force,
                        contact_forces_a, contact_forces_b,
                        non_foot_body_ids,
                        w_alive, w_ball_to_hand, w_pass_direction,
                        w_catch_success, w_ball_height,
                        w_standing, w_upright, w_ground_contact_penalty,
                        termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor) -> Tensor
    """
    Compute cooperative pass-and-catch reward with posture constraints.
    
    Reward components:
    1. alive: Both humanoids standing upright
    2. ball_to_hand: Catcher's hands approaching the ball
    3. pass_direction: Ball velocity pointing toward catcher (ENHANCED with squared alignment)
    4. catch_success: Ball contacted by catcher's hand (CONDITIONAL on standing)
    5. ball_height: Ball at catchable height (0.5m - 1.5m)
    6. standing: NEW - CoM height reward for maintaining standing posture
    7. upright: NEW - Body verticality reward (penalize lying down)
    8. ground_contact_penalty: NEW - Penalty for non-foot ground contact
    
    Args:
        ball_pos: Ball position [N, 3]
        ball_vel: Ball velocity [N, 3]
        root_pos_a/b: Humanoid root position [N, 3]
        root_rot_a/b: Humanoid root rotation quaternion [N, 4]
        height_a/b: Humanoid height [N]
        dist_ball_to_hand_a/b: Distance from ball to closest hand [N]
        ball_contact_force: Contact force on ball [N, 3]
        contact_forces_a/b: Contact forces on humanoid bodies [N, num_bodies, 3]
        non_foot_body_ids: Indices of non-foot bodies [num_non_foot]
        w_*: Reward weights
        termination_heights: Height threshold for falling [N]
        
    Returns:
        Tensor [N]: Total reward
    """
    num_envs = ball_pos.shape[0]
    device = ball_pos.device
    
    # Minimum standing height constant
    MIN_STANDING_HEIGHT = 0.8
    
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    
    # =========== 1. Alive Reward ===========
    # Both humanoids should be standing (above termination height)
    alive_a = (height_a > termination_heights).float()
    alive_b = (height_b > termination_heights).float()
    r_alive = alive_a * alive_b * w_alive
    
    # =========== 2. Ball to Hand Reward ===========
    ball_to_a = torch.norm(ball_pos - root_pos_a, dim=-1)
    ball_to_b = torch.norm(ball_pos - root_pos_b, dim=-1)
    
    # Determine who is closer to ball (passer vs catcher)
    b_is_catcher = (ball_to_a < ball_to_b).float()
    
    # Reward catcher's hand approaching the ball
    catcher_hand_dist = dist_ball_to_hand_b * b_is_catcher + dist_ball_to_hand_a * (1.0 - b_is_catcher)
    r_ball_to_hand = torch.exp(-2.0 * catcher_hand_dist) * w_ball_to_hand
    
    # =========== 3. Pass Direction Reward (ENHANCED) ===========
    # Use SQUARED alignment for steeper reward curve
    ball_speed = torch.norm(ball_vel, dim=-1)
    
    # Direction from ball to catcher
    dir_ball_to_catcher = (root_pos_b * b_is_catcher.unsqueeze(-1) + 
                           root_pos_a * (1.0 - b_is_catcher.unsqueeze(-1))) - ball_pos
    dir_ball_to_catcher_norm = dir_ball_to_catcher / (torch.norm(dir_ball_to_catcher, dim=-1, keepdim=True) + 1e-8)
    
    # Ball velocity direction
    ball_vel_norm = ball_vel / (ball_speed.unsqueeze(-1) + 1e-8)
    
    # Dot product: positive if ball moving toward catcher
    alignment = torch.sum(ball_vel_norm * dir_ball_to_catcher_norm, dim=-1)
    
    # ENHANCED: Use squared alignment for steeper reward curve (more reward for direct passes)
    alignment_squared = torch.pow(torch.clamp(alignment, min=0.0), 2)
    r_pass_direction = alignment_squared * (ball_speed > 0.5).float() * w_pass_direction
    
    # =========== 4. Catch Success Reward (CONDITIONAL on standing) ===========
    ball_has_contact = (torch.norm(ball_contact_force, dim=-1) > 1.0).float()
    catcher_catch = (catcher_hand_dist < 0.15).float()  # Within 15cm of hand
    
    # CONDITIONAL: Only reward catch if catcher is standing properly
    catcher_height = height_b * b_is_catcher + height_a * (1.0 - b_is_catcher)
    catcher_is_standing = (catcher_height > MIN_STANDING_HEIGHT).float()
    r_catch_success = ball_has_contact * catcher_catch * catcher_is_standing * w_catch_success
    
    # =========== 5. Ball Height Reward ===========
    ball_height = ball_pos[:, 2]
    height_in_range = ((ball_height > 0.5) & (ball_height < 1.5)).float()
    r_ball_height = height_in_range * w_ball_height
    
    # =========== 6. Standing Reward (NEW) ===========
    # Reward for maintaining CoM above minimum standing height
    standing_reward_a = torch.clamp((height_a - MIN_STANDING_HEIGHT) / 0.5, 0.0, 1.0)
    standing_reward_b = torch.clamp((height_b - MIN_STANDING_HEIGHT) / 0.5, 0.0, 1.0)
    r_standing = (standing_reward_a + standing_reward_b) * 0.5 * w_standing
    
    # =========== 7. Upright Reward (NEW) ===========
    # Reward for body being vertical (Z-axis of body aligned with world Z)
    # Compute body up vector from rotation quaternion
    # For quaternion q = [x, y, z, w], rotating [0,0,1] gives body's up direction
    # Simplified: use the Z component of rotated up vector
    # quat_rotate formula for [0,0,1]: 
    #   result_x = 2*(x*z + w*y)
    #   result_y = 2*(y*z - w*x)
    #   result_z = 1 - 2*(x*x + y*y)
    x_a, y_a, z_a, w_a = root_rot_a[:, 0], root_rot_a[:, 1], root_rot_a[:, 2], root_rot_a[:, 3]
    x_b, y_b, z_b, w_b = root_rot_b[:, 0], root_rot_b[:, 1], root_rot_b[:, 2], root_rot_b[:, 3]
    
    # Z component of body up vector (1.0 = perfectly upright, 0.0 = horizontal)
    body_up_z_a = 1.0 - 2.0 * (x_a * x_a + y_a * y_a)
    body_up_z_b = 1.0 - 2.0 * (x_b * x_b + y_b * y_b)
    
    # Clamp and reward (0 to 1 range)
    upright_a = torch.clamp(body_up_z_a, 0.0, 1.0)
    upright_b = torch.clamp(body_up_z_b, 0.0, 1.0)
    r_upright = (upright_a + upright_b) * 0.5 * w_upright
    
    # =========== 8. Ground Contact Penalty (NEW) ===========
    # Penalty for non-foot body parts touching ground
    contact_threshold = 10.0  # Force threshold for detecting ground contact
    
    non_foot_contact_a = contact_forces_a[:, non_foot_body_ids, :]
    non_foot_contact_b = contact_forces_b[:, non_foot_body_ids, :]
    
    contact_magnitude_a = torch.norm(non_foot_contact_a, dim=-1)
    contact_magnitude_b = torch.norm(non_foot_contact_b, dim=-1)
    
    # Sum of contact magnitudes above threshold (normalized)
    ground_contact_a = torch.sum(torch.clamp(contact_magnitude_a - contact_threshold, min=0.0), dim=-1) / 100.0
    ground_contact_b = torch.sum(torch.clamp(contact_magnitude_b - contact_threshold, min=0.0), dim=-1) / 100.0
    
    r_ground_contact = (ground_contact_a + ground_contact_b) * w_ground_contact_penalty  # w is negative
    
    # =========== Total Reward ===========
    reward = (r_alive + r_ball_to_hand + r_pass_direction + r_catch_success + 
              r_ball_height + r_standing + r_upright + r_ground_contact)
    
    return reward
