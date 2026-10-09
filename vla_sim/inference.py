"""Shared single-step policy inference for evaluation and visualization."""

import time

import torch

from vla_sim.data import encode_instruction, select_state


def predict_action(model, payload, observation, instruction, device):
    image = torch.from_numpy(observation["image"]).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    state = torch.from_numpy(select_state(observation['state'], payload.get('state_mode', 'full'))).unsqueeze(0).to(device)
    tokens = torch.from_numpy(encode_instruction(instruction, payload["vocab"])).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        action = model(image, state, tokens)[0, 0].cpu().numpy()
    return action, time.perf_counter() - start
