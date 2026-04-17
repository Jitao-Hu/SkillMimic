"""
SkillMimic MultiPlayer Inference Environment.

A visualization-only task that extends ``SkillMimicBallPlay`` with:

  * N role-less humanoids per physical env (configured via ``numPlayers``)
  * one shared basketball (actor index ``N``)
  * one static basket / hoop (actor index ``N + 1``) loaded from
    ``skillmimic/data/assets/urdf/basket.urdf`` with ``fix_base_link=True``

The policy only sees humanoid 0 (the standard ``SkillMimicBallPlay`` observation
pipeline is reused unchanged). Humanoids 1..N-1 are visual copies: at reset
they are placed at configurable X offsets and given the same reference pose as
humanoid 0; every step they receive the same PD target that the LLC produces
for humanoid 0. This keeps the existing LLC checkpoint + train config valid
without any fan-out along a new agent axis.

Actor ordering per env is:

    [humanoid_0, humanoid_1, ..., humanoid_{N-1}, ball, basket]

so that the base ``HumanoidWholeBody`` tensor views that select the first
actor / first ``num_dof`` DOFs / first ``num_bodies`` rigid bodies continue to
point at humanoid 0 without modification. Only the ball / basket views and the
"other humanoid" DOF + root views need custom indexing, done in
``_build_target_tensors`` and ``_build_extra_humanoid_tensors`` below.
"""

import os
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch

from env.tasks.skillmimic import SkillMimicBallPlay


# Distinct RGB colors for humanoids 1..N-1 (humanoid 0 keeps the default green
# from ``HumanoidWholeBody._build_env``). Cycled for N > len(_EXTRA_COLORS).
_EXTRA_COLORS = [
    (0.20, 0.54, 0.85),  # blue
    (0.90, 0.35, 0.10),  # orange
    (0.85, 0.20, 0.54),  # magenta
    (0.95, 0.85, 0.15),  # yellow
    (0.25, 0.80, 0.80),  # cyan
    (0.60, 0.30, 0.85),  # purple
    (0.45, 0.45, 0.45),  # grey
]


class SkillMimicMultiPlayer(SkillMimicBallPlay):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]

        self._num_players = int(env_cfg.get("numPlayers", 2))
        assert self._num_players >= 1, "numPlayers must be >= 1"

        self._player_spacing = float(env_cfg.get("playerSpacing", 1.5))

        basket_pos = env_cfg.get("basketPosition", [5.0, 0.0, 3.05])
        assert len(basket_pos) == 3, "basketPosition must be [x, y, z]"
        self._basket_rim_position = np.array(basket_pos, dtype=np.float32)

        # Rim center in the basket URDF's local frame. Must match basket.urdf.
        self._basket_rim_local_offset = np.array([0.0, 0.225, 3.05], dtype=np.float32)

        # Per-physical-env handle bookkeeping (filled inside _build_env).
        self._player_handles = [[] for _ in range(self._num_players)]
        self._basket_handles = []

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        self._build_extra_humanoid_tensors()
        self._build_basket_tensors()

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------
    def _load_basket_asset(self):
        asset_root = "skillmimic/data/assets/urdf/"
        asset_file = "basket.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._basket_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    # ------------------------------------------------------------------
    # Env construction
    # ------------------------------------------------------------------
    def _create_envs(self, num_envs, spacing, num_per_row):
        # Target (ball) list lives on the parent class; we initialise it here
        # before calling _load_target_asset so the base build_env hook works.
        self._target_handles = []
        self._load_target_asset()
        self._load_basket_asset()

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()

        # --- Humanoid asset load (mirrors HumanoidWholeBody._create_envs) ---
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

        # ``humanoid_handles`` (inherited convention) points at humanoid 0 per
        # env so that base-class helpers like ``_build_key_body_ids_tensor``
        # keep working.
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        num_basket_bodies = self.gym.get_asset_rigid_body_count(self._basket_asset)
        num_basket_shapes = self.gym.get_asset_rigid_shape_count(self._basket_asset)

        max_agg_bodies = (
            self.num_humanoid_bodies * self._num_players + 1 + num_basket_bodies
        )
        max_agg_shapes = (
            self.num_humanoid_shapes * self._num_players + 1 + num_basket_shapes
        )

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if self._pd_control:
            self._build_pd_action_offset_scale()

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        segmentation_id = 0

        # col_filter=1 for humanoids so they DON'T collide with each other
        # (Isaac Gym: actors in same col_group collide only if
        # ``(filter_a & filter_b) == 0``). Ball/basket use col_filter=0, so
        # they still collide with every humanoid.
        humanoid_col_filter = 1

        char_h = 0.89
        spacing = self._player_spacing
        x0 = -0.5 * (self._num_players - 1) * spacing

        for player_idx in range(self._num_players):
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(x0 + player_idx * spacing, 0.0, char_h)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            name = "humanoid_{}".format(player_idx)
            handle = self.gym.create_actor(
                env_ptr, humanoid_asset, start_pose, name,
                col_group, humanoid_col_filter, segmentation_id,
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            if player_idx == 0:
                color = gymapi.Vec3(0.54, 0.85, 0.2)
            else:
                rgb = _EXTRA_COLORS[(player_idx - 1) % len(_EXTRA_COLORS)]
                color = gymapi.Vec3(*rgb)
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, color)

            if self._pd_control:
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            self._player_handles[player_idx].append(handle)
            if player_idx == 0:
                self.humanoid_handles.append(handle)

        self._build_target(env_id, env_ptr)
        self._build_basket(env_id, env_ptr)

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

    def _build_basket(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        rim_world = self._basket_rim_position
        rim_local = self._basket_rim_local_offset
        actor_pos = rim_world - rim_local

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(actor_pos[0]), float(actor_pos[1]), float(actor_pos[2]))
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        basket_handle = self.gym.create_actor(
            env_ptr, self._basket_asset, pose, "basket",
            col_group, col_filter, segmentation_id,
        )
        self._basket_handles.append(basket_handle)

    # ------------------------------------------------------------------
    # Tensor views
    # ------------------------------------------------------------------
    def _build_target_tensors(self):
        """Ball is at actor index ``num_players`` (after all humanoids)."""
        num_actors = self.get_num_actors_per_env()
        ball_idx = self._num_players

        self._target_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., ball_idx, :]

        self._tar_actor_ids = (
            to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32)
            + ball_idx
        )

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        ball_body_idx = self.num_bodies * self._num_players
        self._tar_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., ball_body_idx, :]

        self.init_obj_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)
        self.init_obj_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

    def _build_extra_humanoid_tensors(self):
        """Build views / actor ids for humanoids 1..N-1.

        Humanoid 0 is already covered by the base class views
        (``_humanoid_root_states``, ``_dof_pos``, ...). These extras let us
        reset the other humanoids and apply PD targets to them.
        """
        num_actors = self.get_num_actors_per_env()
        assert num_actors == self._num_players + 2, (
            "Expected {} actors per env (N players + ball + basket), got {}".format(
                self._num_players + 2, num_actors
            )
        )

        # (num_envs, num_actors, 13)
        all_root = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )

        # Root states for every humanoid (num_envs, N, 13).
        self._all_humanoid_root_states = all_root[:, : self._num_players, :]

        # Initial root states — snapshot AFTER base init populated humanoid 0.
        self._initial_all_humanoid_root_states = self._all_humanoid_root_states.clone()
        self._initial_all_humanoid_root_states[..., 7:13] = 0.0

        # Per-humanoid actor ids, flattened (num_envs * N,) in env-major order.
        base = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        offs = torch.arange(self._num_players, device=self.device, dtype=torch.int32)
        self._all_humanoid_actor_ids = (
            base.unsqueeze(1) + offs.unsqueeze(0)
        ).reshape(-1)

        # DOF state view reshaped along a new "player" axis:
        # (num_envs, N, num_dof, 2). Ball / basket contribute no DOFs, so only
        # humanoid DOFs are present in ``_dof_state``.
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        assert dofs_per_env == self._num_players * self.num_dof, (
            "Expected {} DOFs per env, got {}".format(
                self._num_players * self.num_dof, dofs_per_env
            )
        )
        dof_view = self._dof_state.view(
            self.num_envs, self._num_players, self.num_dof, 2
        )
        self._all_dof_pos = dof_view[..., 0]
        self._all_dof_vel = dof_view[..., 1]

    def _build_basket_tensors(self):
        """Basket is a static actor; we just hold its root-state view."""
        num_actors = self.get_num_actors_per_env()
        basket_idx = self._num_players + 1
        self._basket_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., basket_idx, :]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_humanoid(self, env_ids):
        super()._reset_humanoid(env_ids)

        if self._num_players <= 1:
            return

        init_pos = self.init_root_pos[env_ids]
        init_rot = self.init_root_rot[env_ids]
        init_pos_vel = self.init_root_pos_vel[env_ids]
        init_rot_vel = self.init_root_rot_vel[env_ids]

        init_dof = self.init_dof_pos[env_ids]
        init_dof_vel = self.init_dof_pos_vel[env_ids]

        spacing = self._player_spacing
        x0 = -0.5 * (self._num_players - 1) * spacing

        for player_idx in range(1, self._num_players):
            dx = x0 + player_idx * spacing  # relative to humanoid 0 origin
            self._all_humanoid_root_states[env_ids, player_idx, 0] = init_pos[:, 0] + dx
            self._all_humanoid_root_states[env_ids, player_idx, 1] = init_pos[:, 1]
            self._all_humanoid_root_states[env_ids, player_idx, 2] = init_pos[:, 2]
            self._all_humanoid_root_states[env_ids, player_idx, 3:7] = init_rot
            self._all_humanoid_root_states[env_ids, player_idx, 7:10] = init_pos_vel
            self._all_humanoid_root_states[env_ids, player_idx, 10:13] = init_rot_vel

            self._all_dof_pos[env_ids, player_idx, :] = init_dof
            self._all_dof_vel[env_ids, player_idx, :] = init_dof_vel

    def _reset_env_tensors(self, env_ids):
        humanoid_ids_flat = (
            self._all_humanoid_actor_ids.view(self.num_envs, self._num_players)[env_ids]
            .reshape(-1)
        )
        ball_ids = self._tar_actor_ids[env_ids]
        all_root_ids = torch.cat([humanoid_ids_flat, ball_ids])

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(all_root_ids),
            len(all_root_ids),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_ids_flat),
            len(humanoid_ids_flat),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if self._pd_control:
            pd_tar_h0 = self._action_to_pd_targets(self.actions)  # (num_envs, num_dof)
            # Broadcast the same target to every humanoid. ``.expand`` gives a
            # zero-copy view; ``.contiguous()`` is needed so the underlying
            # memory matches the flat shape Isaac Gym expects.
            pd_tar_all = (
                pd_tar_h0.unsqueeze(1)
                .expand(-1, self._num_players, -1)
                .contiguous()
                .view(self.num_envs, self._num_players * self.num_dof)
            )
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(pd_tar_all)
            )
        else:
            forces_h0 = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            forces_all = (
                forces_h0.unsqueeze(1)
                .expand(-1, self._num_players, -1)
                .contiguous()
                .view(self.num_envs, self._num_players * self.num_dof)
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(forces_all)
            )

        # Preserve the viewer-event bookkeeping that parent
        # ``HumanoidWholeBodyWithObject.pre_physics_step`` performs.
        if self.viewer is not None:
            self.evts = list(self.gym.query_viewer_action_events(self.viewer))
        else:
            self.evts = []
