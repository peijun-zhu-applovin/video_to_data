#!/usr/bin/env python
"""Roll out a trained Sharpa-V2D policy and record per-step sim states to NPZ,
for GLB export (agents/scripts/export_rollout_glb.py). Modeled on eval.py.

Records per step (env 0): wrist root pose + finger joint positions per hand,
object root poses, sim fingertip body positions (for FK validation), and the
command's reference object pose + reference fingertip positions (for ghost
overlays). All positions env-frame (env origin subtracted).

Usage (Env B venv, after training):
  .venv/bin/python agents/scripts/record_rollout_states.py --headless \
    --task Sharpa-V2D-v0 --motion_file <processed seq dir> --num_envs 1 \
    --checkpoint <model.pt> --max_steps 400 --out rollout_states.npz
"""
# ruff: noqa: E402
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--out", type=str, required=True)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point"
)
AppLauncher.add_app_launcher_args(parser)
import sys

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import robotic_grounding.tasks  # noqa: F401  register envs
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.motion_file = args_cli.motion_file
    scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
    apply_scene_config(env_cfg, scene_config, use_primitive_urdfs=False)

    # Eval-mode: deterministic reset at frame 0, VOC off (same as eval.py).
    cmd_cfg = env_cfg.commands.dual_hands_object_tracking_command
    cmd_cfg.always_reset_to_first_frame = True
    cmd_cfg.initial_virtual_object_control_curriculum_scale = 0.0
    cmd_cfg.virtual_object_control_decay_steps = 0
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None) or runner.alg.actor_critic

    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("dual_hands_object_tracking_command")
    origin = uenv.scene.env_origins[0].cpu().numpy()

    rec = {k: [] for k in (
        "right_wrist_pos", "right_wrist_wxyz", "right_joint_pos", "right_fingertip_pos",
        "left_wrist_pos", "left_wrist_wxyz", "left_joint_pos", "left_fingertip_pos",
        "object_pos", "object_wxyz", "ref_object_pos", "ref_object_wxyz",
        "ref_right_fingertip_pos", "ref_left_fingertip_pos", "ref_timestep")}

    def snap():
        for side in ("right", "left"):
            robot = getattr(cmd, f"{side}_robot")
            rec[f"{side}_wrist_pos"].append(robot.data.root_link_pos_w[0].cpu().numpy() - origin)
            rec[f"{side}_wrist_wxyz"].append(robot.data.root_link_quat_w[0].cpu().numpy())
            rec[f"{side}_joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
            tips = robot.data.body_pos_w[0, getattr(cmd, f"{side}_fingertip_body_ids").cpu()]
            rec[f"{side}_fingertip_pos"].append(tips.cpu().numpy() - origin)
        rec["object_pos"].append(np.stack([o.data.root_link_pos_w[0].cpu().numpy() - origin for o in cmd.objects]))
        rec["object_wxyz"].append(np.stack([o.data.root_link_quat_w[0].cpu().numpy() for o in cmd.objects]))
        rec["ref_object_pos"].append(cmd.object_body_position_command_e[0].cpu().numpy())
        rec["ref_object_wxyz"].append(cmd.object_body_wxyz_command_e[0].cpu().numpy())
        rec["ref_right_fingertip_pos"].append(cmd.right_hand_fingertip_position_command_e[0].cpu().numpy())
        rec["ref_left_fingertip_pos"].append(cmd.left_hand_fingertip_position_command_e[0].cpu().numpy())
        rec["ref_timestep"].append(int(cmd.timestep_counter[0].item()))

    obs = env.get_observations()
    snap()
    for _ in range(args_cli.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        snap()
        import isaaclab.utils.math as math_utils
        rdiff = torch.norm(cmd.right_hand_wrist_pose_command_e[:, :3] - cmd.right_hand_wrist_position_e, dim=-1)[0]
        ldiff = torch.norm(cmd.left_hand_wrist_pose_command_e[:, :3] - cmd.left_hand_wrist_position_e, dim=-1)[0]
        odiff = torch.norm(cmd.object_body_position_command_e - cmd.object_position_e, dim=-1)[0]
        oang = math_utils.quat_error_magnitude(cmd.object_orientation_e, cmd.object_body_wxyz_command_e)[0] \
            if hasattr(cmd, 'object_orientation_e') else torch.tensor(float('nan'))
        print(f"[diag] step={len(rec['ref_timestep'])-1} tc={int(cmd.timestep_counter[0])} "
              f"rwrist={rdiff.item():.3f} lwrist={ldiff.item():.3f} "
              f"objpos={[round(v,3) for v in odiff.tolist()]} objang={oang.tolist() if oang.dim() else round(float(oang),3)}")
        if bool(dones[0]):
            tm = uenv.termination_manager
            fired = {n: bool(tm.get_term(n)[0].item()) for n in tm.active_terms}
            print(f"[recorder] done at step {len(rec['ref_timestep'])-1}; fired terms: "
                  f"{[n for n, v in fired.items() if v]}")
            if os.environ.get("STOP_ON_DONE", "1") == "1":
                break

    meta = {
        "right_joint_names": list(cmd.right_robot.data.joint_names),
        "left_joint_names": list(cmd.left_robot.data.joint_names),
        "right_fingertip_body_names": list(cmd.right_fingertip_body_names),
        "left_fingertip_body_names": list(cmd.left_fingertip_body_names),
        "object_body_names": list(cmd.cfg.object_body_names),
        "step_dt": float(uenv.step_dt),
        "motion_file": args_cli.motion_file,
        "checkpoint": args_cli.checkpoint,
    }
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(args_cli.out, meta=np.array([__import__("json").dumps(meta)]), **arrays)
    print(f"[recorder] wrote {args_cli.out}: {arrays['right_wrist_pos'].shape[0]} steps")
    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
