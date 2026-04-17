"""
SkillMimic MultiPlayer Inference Environment.

Extends :class:`HRLScoringLayup` with N role-less humanoids per physical env
sharing a single basketball and a drawn scoring goal (the original
``HRLScoringLayup._draw_task`` visualisation — no URDF hoop is built).

Architecture
------------

rl_games sees ``P * N`` "virtual envs" where ``P`` is ``numEnvs`` from the
config and ``N`` is ``numPlayers``. Each virtual env corresponds to ONE
humanoid, so the HRL + LLC policy is evaluated once per humanoid per step
and every humanoid independently picks its own discrete skill and LLC action
(they contest the ball rather than moving in lockstep).

Isaac Gym only has ``P`` physical envs; each one contains ``N`` humanoid
actors (actor ids ``0..N-1``) and one ball actor (actor id ``N``). Base-class
tensor views produced by ``HumanoidWholeBody.__init__`` assume a single
humanoid per env, so after ``super().__init__()`` (with ``num_envs = P``) we
rebuild the observation/reset tensors as ``(P*N, ...)`` and bump
``self.num_envs`` to ``V = P * N`` before rl_games wraps the task.

Per-physical-env quantities (the ball, contact forces on the ball, the
scoring goal) are broadcast to every virtual env so each humanoid sees the
same ball and target. Humanoid resets modify a ``(V, ...)`` buffer, which is
pushed back into the underlying Isaac Gym tensor via a non-contiguous 3D
view of ``_root_states`` / ``_dof_state`` before
``set_actor_root_state_tensor_indexed`` is called.

Reset is synchronised per physical env: if any humanoid triggers a reset
(fall / episode timeout) the whole physical env resets together (all ``N``
humanoids + the ball), which keeps the shared ball in a sensible state.
"""

import math
import os
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch

from env.tasks.hrl_scoring_layup import HRLScoringLayup


_EXTRA_COLORS = [
    (0.20, 0.54, 0.85),  # blue
    (0.90, 0.35, 0.10),  # orange
    (0.85, 0.20, 0.54),  # magenta
    (0.95, 0.85, 0.15),  # yellow
    (0.25, 0.80, 0.80),  # cyan
    (0.60, 0.30, 0.85),  # purple
    (0.45, 0.45, 0.45),  # grey
]


def _player_x_offset(player_idx: int, spacing: float) -> float:
    """Alternating left/right spacing so no two humanoids overlap.

    Layout for N=3, spacing=1.5: [0, +1.5, -1.5].
    For N=5: [0, +1.5, -1.5, +3.0, -3.0]. Keeps humanoid 0 at the reference
    pose (the value used to set ``init_root_pos`` from the motion clip) and
    spreads the rest symmetrically on either side.
    """
    if player_idx == 0:
        return 0.0
    step = (player_idx + 1) // 2  # 1,1,2,2,3,3,...
    sign = 1.0 if (player_idx % 2 == 1) else -1.0
    return sign * step * spacing


class SkillMimicMultiPlayer(HRLScoringLayup):
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]

        self._num_players = int(env_cfg.get("numPlayers", 3))
        assert self._num_players >= 1, "numPlayers must be >= 1"
        self._player_spacing = float(env_cfg.get("playerSpacing", 1.5))

        # cfg["env"]["numEnvs"] is the number of PHYSICAL scenes. We keep it
        # unchanged during the parent init so Isaac Gym tensors are sized for
        # P envs; we expand to V = P * N virtual envs after the parent init
        # finishes, before rl_games wraps the task.
        self._num_physical_envs = int(env_cfg["numEnvs"])

        # Per-physical-env handle bookkeeping (populated in _build_env).
        self._player_handles = [[] for _ in range(self._num_players)]

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        # Re-shape everything for P * N virtual envs, one per humanoid.
        self._expand_to_virtual_envs()

    # ------------------------------------------------------------------
    # Asset + env construction (override HumanoidWholeBodyWithObject)
    # ------------------------------------------------------------------
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()
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

        # ``humanoid_handles`` holds humanoid 0 per env (kept for base-class
        # helpers like ``_build_key_body_ids_tensor``).
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies * self._num_players + 1
        max_agg_shapes = self.num_humanoid_shapes * self._num_players + 1

        for i in range(num_envs):
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

        # ``col_filter=1`` on every humanoid so humanoids in the same
        # col_group don't collide with each other (AND of their filters is
        # non-zero → Isaac Gym skips collision). Ball is col_filter=0 so it
        # still collides with each humanoid.
        humanoid_col_filter = 1

        char_h = 0.89
        spacing = self._player_spacing

        for player_idx in range(self._num_players):
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(_player_x_offset(player_idx, spacing), 0.0, char_h)
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

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

    # ------------------------------------------------------------------
    # Tensor views for the BALL (ball is at actor index N, not 1)
    # ------------------------------------------------------------------
    def _build_target_tensors(self):
        P = self._num_physical_envs
        num_actors = self.get_num_actors_per_env()  # = N + 1
        ball_idx = self._num_players

        root_view = self._root_states.view(P, num_actors, self._root_states.shape[-1])
        # Per-physical-env ball root state view ((P, 13), writable into _root_states).
        self._ball_root_view_phys = root_view[:, ball_idx, :]

        self._tar_actor_ids = (
            to_torch(num_actors * np.arange(P), device=self.device, dtype=torch.int32)
            + ball_idx
        )

        bodies_per_env = self._rigid_body_state.shape[0] // P
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        ball_body_idx = self.num_bodies * self._num_players
        # (P, 3) non-contig view of ball contact forces.
        self._ball_contact_view_phys = contact_force_tensor.view(P, bodies_per_env, 3)[:, ball_body_idx, :]

        # These are placeholder (P, ...) tensors; _expand_to_virtual_envs
        # rebinds them to (V, ...) sizes. They get populated per virtual env
        # by _reset_random_ref_state_init (which reads motion data).
        self.init_obj_pos = torch.zeros([P, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([P, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float
        ).repeat(P, 1)
        self.init_obj_rot_vel = torch.zeros([P, 3], device=self.device, dtype=torch.float)

        # Base-class facing code reads ``self._target_states``; after expand
        # this is rebound to a (V, 13) buffer. During super init (parent of
        # HRLScoringLayup) we give it a (P, 13) stand-in which coincides with
        # the physical ball view so any intermediate code keeps working.
        self._target_states = self._ball_root_view_phys
        self._tar_contact_forces = self._ball_contact_view_phys

    # ------------------------------------------------------------------
    # Expand (P, ...) tensors -> (V = P * N, ...) for rl_games
    # ------------------------------------------------------------------
    def _expand_to_virtual_envs(self):
        P = self._num_physical_envs
        N = self._num_players
        V = P * N
        num_actors = self.get_num_actors_per_env()  # = N + 1

        # ---- 3D VIEWS of physics tensors (writes propagate) ----
        root_view = self._root_states.view(P, num_actors, 13)
        self._humanoid_root_view = root_view[:, :N, :]          # (P, N, 13)
        self._ball_root_view = root_view[:, N, :]               # (P, 13)
        self._ball_root_view_phys = self._ball_root_view

        dof_view = self._dof_state.view(P, N, self.num_dof, 2)
        self._dof_pos_view = dof_view[..., 0]                   # (P, N, num_dof)
        self._dof_vel_view = dof_view[..., 1]

        bodies_per_env = self._rigid_body_state.shape[0] // P   # = N * num_bodies + 1
        rb_view = self._rigid_body_state.view(P, bodies_per_env, 13)
        self._humanoid_rb_view = rb_view[:, : N * self.num_bodies, :].view(
            P, N, self.num_bodies, 13
        )
        self._ball_rb_view = rb_view[:, N * self.num_bodies, :]  # (P, 13)

        # Contact forces
        cf_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        cf_tensor = gymtorch.wrap_tensor(cf_tensor)
        cf_view = cf_tensor.view(P, bodies_per_env, 3)
        self._humanoid_cf_view = cf_view[:, : N * self.num_bodies, :].view(
            P, N, self.num_bodies, 3
        )
        self._ball_cf_view = cf_view[:, N * self.num_bodies, :]
        self._ball_contact_view_phys = self._ball_cf_view

        # ---- (V, ...) BUFFERS used by obs/reward/reset code ----
        # These are copies that get synced from the 3D views every refresh;
        # humanoid writes during reset push back through the 3D views.
        self._humanoid_root_states = self._humanoid_root_view.reshape(V, 13).contiguous()
        self._dof_pos = self._dof_pos_view.reshape(V, self.num_dof).contiguous()
        self._dof_vel = self._dof_vel_view.reshape(V, self.num_dof).contiguous()

        self._rigid_body_pos = self._humanoid_rb_view[..., 0:3].reshape(V, self.num_bodies, 3).contiguous()
        self._rigid_body_rot = self._humanoid_rb_view[..., 3:7].reshape(V, self.num_bodies, 4).contiguous()
        self._rigid_body_vel = self._humanoid_rb_view[..., 7:10].reshape(V, self.num_bodies, 3).contiguous()
        self._rigid_body_ang_vel = self._humanoid_rb_view[..., 10:13].reshape(V, self.num_bodies, 3).contiguous()
        self._contact_forces = self._humanoid_cf_view.reshape(V, self.num_bodies, 3).contiguous()

        # Per-virtual-env ball state buffers (broadcast of the (P, ...) ball)
        self._target_states = torch.zeros(V, 13, device=self.device, dtype=torch.float)
        self._tar_contact_forces = torch.zeros(V, 3, device=self.device, dtype=torch.float)

        # ---- Init state buffers (V, ...) ----
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0.0
        self.init_root_pos = self._initial_humanoid_root_states[:, 0:3].clone()
        self.init_root_rot = torch.zeros(V, 4, device=self.device, dtype=torch.float)
        self.init_root_pos_vel = torch.zeros(V, 3, device=self.device, dtype=torch.float)
        self.init_root_rot_vel = torch.zeros(V, 3, device=self.device, dtype=torch.float)
        self.init_dof_pos = torch.zeros(V, self.num_dof, device=self.device, dtype=torch.float)
        self.init_dof_pos_vel = torch.zeros(V, self.num_dof, device=self.device, dtype=torch.float)

        self.init_obj_pos = torch.zeros(V, 3, device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros(V, 3, device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float
        ).repeat(V, 1)
        self.init_obj_rot_vel = torch.zeros(V, 3, device=self.device, dtype=torch.float)

        # Goal / reached_target are per virtual env (each humanoid has its
        # own goal, randomised around its own spawn). That lets all N
        # humanoids scramble for the ball instead of aiming at a single
        # point.
        self._goal_position = torch.zeros(V, 2, device=self.device, dtype=torch.float)
        self.reached_target = torch.zeros(V, device=self.device, dtype=torch.bool)

        # ---- Actor ID tensors ----
        phys_idx = torch.arange(P, device=self.device, dtype=torch.int32).unsqueeze(1)
        player_idx = torch.arange(N, device=self.device, dtype=torch.int32).unsqueeze(0)
        self._humanoid_actor_ids = (phys_idx * num_actors + player_idx).reshape(V)
        self._tar_actor_ids = (
            torch.arange(P, device=self.device, dtype=torch.int32) * num_actors + N
        )

        # ---- Per-virtual-env buffers (obs / reward / reset / progress) ----
        num_obs = self.obs_buf.shape[1]
        self.obs_buf = torch.zeros((V, num_obs), device=self.device, dtype=torch.float)
        if self.num_states > 0:
            self.states_buf = torch.zeros((V, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(V, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(V, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(V, device=self.device, dtype=torch.long)
        self._terminate_buf = torch.zeros(V, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(V, device=self.device, dtype=torch.long)

        # ---- Bump num_envs so rl_games / VecTaskPythonWrapper see V ----
        self.num_envs = V

        # ---- Reload motion handler with V envs ----
        # HRLScoringLayup._load_motion built _motion_data with P envs;
        # its internal buffers (envid2motid etc.) need to be V-sized.
        self._load_motion(self.motion_file)

        # Pre-compute per-player X offsets for _reset_random_ref_state_init.
        self._player_x_offsets = torch.tensor(
            [_player_x_offset(i, self._player_spacing) for i in range(N)],
            device=self.device,
            dtype=torch.float,
        )

        # Freshly pull physics values into the (V, ...) buffers.
        self._sync_virtual_from_physics()

    # ------------------------------------------------------------------
    # Physics <-> buffer sync
    # ------------------------------------------------------------------
    def _sync_virtual_from_physics(self):
        """Copy physics tensors into the (V, ...) buffers read by obs code."""
        P = self._num_physical_envs
        N = self._num_players
        V = self.num_envs

        self._humanoid_root_states.copy_(self._humanoid_root_view.reshape(V, 13))
        self._dof_pos.copy_(self._dof_pos_view.reshape(V, self.num_dof))
        self._dof_vel.copy_(self._dof_vel_view.reshape(V, self.num_dof))
        self._rigid_body_pos.copy_(self._humanoid_rb_view[..., 0:3].reshape(V, self.num_bodies, 3))
        self._rigid_body_rot.copy_(self._humanoid_rb_view[..., 3:7].reshape(V, self.num_bodies, 4))
        self._rigid_body_vel.copy_(self._humanoid_rb_view[..., 7:10].reshape(V, self.num_bodies, 3))
        self._rigid_body_ang_vel.copy_(self._humanoid_rb_view[..., 10:13].reshape(V, self.num_bodies, 3))
        self._contact_forces.copy_(self._humanoid_cf_view.reshape(V, self.num_bodies, 3))

        # Broadcast shared ball state (P, 13) -> (V, 13).
        self._target_states.copy_(
            self._ball_root_view.unsqueeze(1).expand(P, N, 13).reshape(V, 13)
        )
        self._tar_contact_forces.copy_(
            self._ball_cf_view.unsqueeze(1).expand(P, N, 3).reshape(V, 3)
        )

    def _push_humanoid_to_physics(self, env_ids):
        """Write the (V, ...) humanoid root + DOF buffers back into physics for env_ids.

        env_ids: (M,) long tensor of virtual env indices.
        """
        N = self._num_players
        phys = (env_ids // N).long()
        player = (env_ids % N).long()
        self._humanoid_root_view[phys, player, :] = self._humanoid_root_states[env_ids, :]
        self._dof_pos_view[phys, player, :] = self._dof_pos[env_ids, :]
        self._dof_vel_view[phys, player, :] = self._dof_vel[env_ids, :]

    # ------------------------------------------------------------------
    # Refresh hook: keep virtual buffers in sync with physics
    # ------------------------------------------------------------------
    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        # Before expand finishes, _humanoid_root_view doesn't exist yet; the
        # base class refresh path is all we need during super init.
        if getattr(self, "_humanoid_root_view", None) is not None and getattr(
            self, "_humanoid_root_states", None
        ) is not None and self._humanoid_root_states.shape[0] == self.num_envs:
            self._sync_virtual_from_physics()

    # ------------------------------------------------------------------
    # Reset path
    # ------------------------------------------------------------------
    def _reset_target(self, env_ids):
        """Reset the shared ball per physical env.

        Reset is synchronised so ``env_ids`` contains all N virtual envs for
        every affected physical env; we pick ``player_idx == 0`` as the
        representative to supply ``init_obj_*`` values.
        """
        if len(env_ids) == 0:
            return
        N = self._num_players
        reps = env_ids[(env_ids % N) == 0]
        if len(reps) == 0:
            return
        phys_ids = (reps // N).long()

        self._ball_root_view[phys_ids, 0:3] = self.init_obj_pos[reps]
        self._ball_root_view[phys_ids, 3:7] = self.init_obj_rot[reps]
        self._ball_root_view[phys_ids, 7:10] = self.init_obj_pos_vel[reps]
        self._ball_root_view[phys_ids, 10:13] = self.init_obj_rot_vel[reps]

    def _reset_env_tensors(self, env_ids):
        if len(env_ids) == 0:
            return

        # 1) push humanoid buffer writes back into _root_states / _dof_state
        #    via the 3D views, so set_*_indexed picks up our changes.
        self._push_humanoid_to_physics(env_ids)

        N = self._num_players
        humanoid_actor_ids = self._humanoid_actor_ids[env_ids]
        # Dedup ball actor ids: one per affected physical env.
        phys_ids = torch.unique((env_ids // N).long())
        ball_actor_ids = self._tar_actor_ids[phys_ids.to(torch.int32)]
        all_actor_ids = torch.cat([humanoid_actor_ids, ball_actor_ids])

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(all_actor_ids),
            len(all_actor_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_actor_ids),
            len(humanoid_actor_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    # ------------------------------------------------------------------
    # Reset motion init: re-anchor players 1..N-1 to player 0's spawn + X
    # offset so the N humanoids appear spatially clustered around a single
    # shared ball instead of teleporting to wherever their individually
    # sampled motion clip happens to start.
    # ------------------------------------------------------------------
    def _reset_random_ref_state_init(self, env_ids):
        super()._reset_random_ref_state_init(env_ids)
        self._reanchor_humanoid_spawns(env_ids)

    def _reset_deterministic_ref_state_init(self, env_ids):
        super()._reset_deterministic_ref_state_init(env_ids)
        self._reanchor_humanoid_spawns(env_ids)

    def _reanchor_humanoid_spawns(self, env_ids):
        if self._num_players <= 1 or len(env_ids) == 0:
            return
        N = self._num_players
        phys = (env_ids // N).long()
        player = (env_ids % N).long()
        # Virtual id of player 0 in each virtual env's phys env.
        player0_ids = phys * N
        # Anchor XY to player 0's sampled root pos + per-player X offset.
        anchor_pos = self.init_root_pos[player0_ids].clone()
        dx = self._player_x_offsets[player]
        new_pos = anchor_pos
        new_pos[:, 0] = anchor_pos[:, 0] + dx
        # Keep each humanoid's own Z (so they still land on feet given their
        # own sampled motion frame).
        new_pos[:, 2] = self.init_root_pos[env_ids, 2]
        self.init_root_pos[env_ids] = new_pos

    # ------------------------------------------------------------------
    # Reset synchronization: if any humanoid in a phys env resets, all do.
    # ------------------------------------------------------------------
    def _synchronize_resets_per_physical_env(self, buf):
        """Expand per-humanoid reset/terminate flags to whole physical envs."""
        N = self._num_players
        phys = buf.view(-1, N).any(dim=1)
        return phys.unsqueeze(1).expand(-1, N).reshape(-1).to(buf.dtype)

    def _compute_reset(self):
        super()._compute_reset()
        if self._num_players > 1:
            self.reset_buf[:] = self._synchronize_resets_per_physical_env(self.reset_buf.bool())
            self._terminate_buf[:] = self._synchronize_resets_per_physical_env(self._terminate_buf.bool())

    # ------------------------------------------------------------------
    # Reset goal: override HRLScoringLayup's per-env goal assignment so that
    # all N humanoids in a physical env share the same goal (one scoring
    # target per phys env, around humanoid 0's current position).
    # ------------------------------------------------------------------
    def _reset_envs(self, env_ids):
        # Run HumanoidWholeBody._reset_envs (reset_actors -> reset_env_tensors
        # -> refresh -> compute_observations) directly so we can handle the
        # goal logic per physical env afterwards.
        if len(env_ids) == 0:
            return

        self._reset_actors(env_ids)
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

        N = self._num_players
        device = self._goal_position.device
        env_ids_dev = env_ids.to(device=device, dtype=torch.long)
        reps = env_ids_dev[(env_ids_dev % N) == 0]
        if len(reps) == 0:
            return
        phys_ids = (reps // N).long()
        n_phys = phys_ids.shape[0]

        d = torch.rand(n_phys, device=device) * 6 + 2
        theta = torch.rand(n_phys, device=device) * (2 * math.pi)
        base_x = self._humanoid_root_states[reps, 0]
        base_y = self._humanoid_root_states[reps, 1]
        goal_x = base_x + torch.sin(theta) * d
        goal_y = base_y + torch.cos(theta) * d

        # Broadcast: all N virtual envs of each affected physical env share the same goal.
        players = torch.arange(N, device=device, dtype=torch.long)
        virt_ids = (phys_ids.unsqueeze(1) * N + players.unsqueeze(0)).reshape(-1)
        goal_x_virt = goal_x.unsqueeze(1).expand(n_phys, N).reshape(-1)
        goal_y_virt = goal_y.unsqueeze(1).expand(n_phys, N).reshape(-1)
        self._goal_position[virt_ids, 0] = goal_x_virt
        self._goal_position[virt_ids, 1] = goal_y_virt
        self.reached_target[virt_ids] = False

    # ------------------------------------------------------------------
    # Draw the scoring goal for each physical env (reuse HRLScoringLayup's
    # line visualisation — this IS the "hoop" the original code renders).
    # ------------------------------------------------------------------
    def _draw_task(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        N = self._num_players
        red = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        green = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        for phys_id, env_ptr in enumerate(self.envs):
            # All N virtual envs of a phys env share the same goal after
            # our _reset_envs override; read from player 0.
            v = phys_id * N
            color = green if self.reached_target[v].item() else red
            gx = float(self._goal_position[v, 0])
            gy = float(self._goal_position[v, 1])
            v1 = np.array([gx - 0.25, gy - 0.25, 2.6, gx + 0.25, gy + 0.25, 2.6], dtype=np.float32).reshape(1, 6)
            v2 = np.array([gx - 0.25, gy + 0.25, 2.6, gx + 0.25, gy - 0.25, 2.6], dtype=np.float32).reshape(1, 6)
            self.gym.add_lines(self.viewer, env_ptr, 1, v1, color)
            self.gym.add_lines(self.viewer, env_ptr, 1, v2, color)
