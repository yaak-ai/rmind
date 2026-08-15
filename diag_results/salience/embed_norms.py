"""Salience measurement: how loud is the max-speed token in the trunk's input?

Compares per-token L2 norms of the three token families that make up each frame
block -- projected patches, the speed token, the max-speed token -- on REAL val
batches, plus the readout's sensitivity to each. Answers: is the conditioning
signal numerically negligible in the residual stream, or is it audible and
ignored by the action head?
"""

from __future__ import annotations

import argparse

import torch

from rmind.models.patch_policy import PatchPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--batch-cache", required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    batches = torch.load(args.batch_cache, weights_only=False)[: args.batches]

    def to_dev(node):
        if isinstance(node, dict):
            return {k: to_dev(v) for k, v in node.items()}
        if isinstance(node, torch.Tensor):
            return node.to(device)
        return node

    def slice_tree(node, s, e):
        if isinstance(node, dict):
            return {k: slice_tree(v, s, e) for k, v in node.items()}
        if isinstance(node, torch.Tensor):
            return node[s:e]
        return node

    for spec in args.ckpt:
        _name, path = spec.split("=", 1)
        model = PatchPolicy.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        model.sample_codes = False
        model = model.to(device).eval()

        # --- 1. Static weight norms (independent of any batch) ---
        emb = model.max_speed_embedding.weight.detach()  # (13, d)
        model.speed_embedding.weight.detach()  # (speed_bins, d)
        {i: float(emb[i].norm()) for i in range(emb.shape[0])}
        # spread between classes: if ~0 the token carries no distinguishable signal
        torch.cdist(emb, emb)
        torch.triu_indices(emb.shape[0], emb.shape[0], offset=1)

        # --- 2. Live token norms inside the trunk input ---
        pn, sn, mn = [], [], []
        for batch in batches:
            bsz = batch["data"]["meta/VehicleMotion/speed"].shape[0]
            for s in range(0, bsz, args.micro_batch):
                d = to_dev(slice_tree(batch, s, s + args.micro_batch))
                with torch.no_grad():
                    inputs = model.input_transform(d)
                    img = model._get(inputs, model.image)
                    feats = model.image_encoder(img)
                    goal = model.goal_encoder.encode(
                        model._get(inputs, model.waypoints)
                    )
                    if goal.ndim == 3:
                        goal = goal.unsqueeze(2).expand(-1, -1, feats.shape[2], -1)
                    patches = model.patch_projection(torch.cat([feats, goal], dim=-1))
                    speed = model._get(inputs, model.speed)
                    stok = model.speed_embedding(model.speed_tokenizer(speed))
                    mtok, _ = model._max_speed_token(inputs, reference=speed)
                pn.append(patches.norm(dim=-1).flatten().cpu())
                sn.append(stok.norm(dim=-1).flatten().cpu())
                mn.append(mtok.norm(dim=-1).flatten().cpu())
                del d
            torch.cuda.empty_cache()

        torch.cat(pn)
        torch.cat(sn)
        torch.cat(mn)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
