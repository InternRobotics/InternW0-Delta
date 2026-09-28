"""One model and one stateful simulator session per WebSocket server."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_config(seed: int = 0):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="sim_robodojo", overrides=[f"seed={seed}"])
    OmegaConf.resolve(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    cfg = load_config(args.seed)
    from eval.robodojo.bootstrap import setup_policy_paths

    setup_policy_paths()
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from eval.robodojo.policy import Model

    model = Model({
        "config": cfg,
        "action_type": "joint",
        "env_cfg_type": "arx_x5",
        "action_horizon": cfg.EVALUATION.action_horizon,
        "replan_steps": cfg.EVALUATION.replan_steps,
    })
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
