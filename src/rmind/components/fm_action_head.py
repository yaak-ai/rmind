"""Action heads for the WTA-vs-ACT-prior study (shared by the toy/BlockPush and Ant pipelines).

Every head maps trunk features f (n, F) [+ a conditioning vector c (n, Cd) for the CVAE
encoders, = the current low-dim observation, as ACT's encoder sees qpos] to an action chunk
of D = W*A values, in normalized action units.

  l1       one regressor, L1 loss (deterministic baseline)
  wta      K regressors (each the WHOLE chunk) + classifier p(k|obs).
           winner = argmin_k mean|a - pred_k| (exactly the l1 loss of head k; no_grad);
           relaxed WTA: winner weight 1-eps, every other head eps/(K-1); eps annealed
           geometrically eps_start -> eps_end over the first `eps_anneal` of training;
           classifier: CE on the winner index. Output biases k-means initialised.
  cvae     ACT: encoder q(z|a,c) Gaussian, prior N(0,I), loss L1 + beta*KL (ACT: beta=10,
           KL summed over z dims, mean over batch). Inference z = 0.
  cvae_lp  same, but a learned Gaussian prior p(z|obs) from the trunk; KL(q || p_obs).
           Inference: prior mean ("argmax") or z ~ p(z|obs) ("sample").
  cvae_cat discretized z: categorical posterior q(k|a,c) over K modes, decoder(f, onehot k),
           learned prior p(k|obs) (the "classifier"); recon is the exact expectation over q
           (all K decoder outputs are computed), + beta*KL(q || p). Inference: argmax / sample k.
  fm       flow matching (rectified flow / linear-interpolation CFM, pi0-style): velocity MLP
           v(x_t, t, f) over the whole chunk; x_t = (1-t) x0 + t a, x0 ~ N(0,I), t ~ U(0,1),
           loss MSE(v, a - x0). Inference: Euler from t=0 to 1 in `fm_steps` steps, trunk features
           computed once. 'sample': x0 ~ N(0,I); 'argmax': x0 = 0 (deterministic, noise-free start).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("l1", "wta", "cvae", "cvae_lp", "cvae_cat", "fm")


def mlp(i, hidden, o):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def time_embed(t, dim=64):
    """sinusoidal embedding of the flow time t in [0,1] -> (n, dim)"""
    half = dim // 2
    fr = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    x = t[:, None] * 1000.0 * fr[None]
    return torch.cat([x.sin(), x.cos()], -1)


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(-1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    e = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e = e + m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(e)


class ActionHead(nn.Module):
    def __init__(self, kind, feat_dim, cond_dim, out_dim, K=1, zdim=8, beta=10.0,
                 eps_start=0.1, eps_end=0.01, eps_anneal=0.5, ce_weight=1.0, hidden=1024,
                 enc_hidden=256):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.D, self.K, self.zdim, self.beta = kind, out_dim, K, zdim, beta
        self.eps_start, self.eps_end, self.eps_anneal, self.ce_weight = eps_start, eps_end, eps_anneal, ce_weight
        H = [hidden, hidden]
        self.register_buffer("progress", torch.zeros(()))  # training progress in [0,1], set by the trainer
        self.fm_steps, self.tdim = 10, 64
        if kind == "l1":
            self.reg = mlp(feat_dim, H, out_dim)
        elif kind == "fm":
            self.vel = mlp(feat_dim + out_dim + self.tdim, H, out_dim)
        elif kind == "wta":
            self.reg = mlp(feat_dim, H, K * out_dim)
            self.cls = mlp(feat_dim, H, K)
        else:
            zin = K if kind == "cvae_cat" else zdim
            self.dec = mlp(feat_dim + zin, H, out_dim)
            qout = K if kind == "cvae_cat" else 2 * zdim
            self.enc = mlp(cond_dim + out_dim, [enc_hidden, enc_hidden], qout)
            if kind == "cvae_lp":
                self.prior = mlp(feat_dim, H, 2 * zdim)
            if kind == "cvae_cat":
                self.prior = mlp(feat_dim, H, K)

    # ------------------------------------------------------------------ helpers
    @property
    def discrete(self):
        return self.kind in ("wta", "cvae_cat")

    def eps(self):
        p = min(float(self.progress) / max(self.eps_anneal, 1e-9), 1.0)
        return self.eps_start * (self.eps_end / self.eps_start) ** p

    @torch.no_grad()
    def kmeans_init(self, targets, iters=50, seed=0):
        """targets (n, D) normalized: set head k's output bias to k-means centroid k."""
        if self.kind != "wta" or self.K == 1:
            return None
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = targets[torch.randperm(len(targets), generator=g)[:200000].to(targets.device)].float()
        c = x[torch.randperm(len(x), generator=g)[: self.K].to(x.device)].clone()
        for _ in range(iters):
            a = torch.cdist(x, c).argmin(1)
            for k in range(self.K):
                if (a == k).any():
                    c[k] = x[a == k].mean(0)
        last = self.reg[-1]
        last.bias.data.copy_(c.reshape(-1).to(last.bias.dtype))
        return c

    def all_heads(self, f):
        """(n, K, D) outputs of all discrete modes (wta heads / cvae_cat decoder per k)."""
        n = f.shape[0]
        if self.kind == "wta":
            return self.reg(f).view(n, self.K, self.D)
        oh = torch.eye(self.K, device=f.device, dtype=f.dtype)
        fz = torch.cat([f[:, None].expand(-1, self.K, -1), oh[None].expand(n, -1, -1)], -1)
        return self.dec(fz)

    def mode_logits(self, f):
        return self.cls(f) if self.kind == "wta" else self.prior(f)

    # ------------------------------------------------------------------ training
    def loss(self, f, c, target, mask=None):
        """f (n,F), c (n,Cd), target (n,D); mask (n,D) or None (1 = real action, 0 = padding past
        the episode end, excluded from the loss; None = every slot counts, the default).
        Returns loss, dict of detached 0-d tensors."""
        d = {}
        if mask is not None:
            assert self.kind in ("l1", "fm", "cvae", "cvae_lp"), "pad mask: continuous heads only"

            def mmean(x):
                return (x * mask).sum() / mask.sum().clamp_min(1.0)
        if self.kind == "l1":
            pred = self.reg(f)
            l = (pred - target).abs().mean() if mask is None else mmean((pred - target).abs())
            return l, {"l1": l.detach(), "l1_policy": l.detach()}
        if self.kind == "fm":
            x0 = torch.randn_like(target)
            t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            xt = (1 - t[:, None]) * x0 + t[:, None] * target
            v = self.velocity(f, xt, t)
            err = (v - (target - x0)).pow(2)
            l = err.mean() if mask is None else mmean(err)
            d = {"fm_mse": l.detach()}
            if not self.training:
                with torch.no_grad():
                    d["l1_policy"] = (self.act(f, "sample")[0] - target).abs().mean()
            return l, d
        if self.discrete:
            preds = self.all_heads(f)  # (n,K,D)
            per = (preds - target[:, None]).abs().mean(-1)  # (n,K) = l1 of each head
            win = per.detach().argmin(1)
            logits = self.mode_logits(f)
            with torch.no_grad():
                pick = logits.argmax(-1)
                d["l1_win"] = per.gather(1, win[:, None]).mean()
                d["l1_policy"] = per.gather(1, pick[:, None]).mean()  # argmax-mode action
                d["cls_acc"] = (pick == win).float().mean()
                freq = torch.bincount(win, minlength=self.K).float() / len(win)
                for k in range(self.K):
                    d[f"win_freq/{k}"] = freq[k]
                d["win_freq_max"] = freq.max()
            if self.kind == "wta":
                eps = self.eps() if self.K > 1 else 0.0
                oh = F.one_hot(win, self.K).float()
                w = oh * (1 - eps) + (1 - oh) * (eps / max(self.K - 1, 1))
                reg = (w * per).sum(1).mean()
                ce = F.cross_entropy(logits, win)
                d.update(reg=reg.detach(), ce=ce.detach(), eps=torch.tensor(eps))
                return reg + self.ce_weight * ce, d
            # cvae_cat
            q_logits = self.enc(torch.cat([c, target], -1))
            q = q_logits.softmax(-1)
            logq, logp = q_logits.log_softmax(-1), logits.log_softmax(-1)
            recon = (q * per).sum(1).mean()
            kl = (q * (logq - logp)).sum(1).mean()
            with torch.no_grad():
                d["ce"] = F.cross_entropy(logits, win)
                d["q_H"] = -(q * logq).sum(1).mean()
                d["q_agree_win"] = (q.argmax(-1) == win).float().mean()
            d.update(recon=recon.detach(), kl=kl.detach())
            return recon + self.beta * kl, d
        # gaussian cvae
        mu, logvar = self.enc(torch.cat([c, target], -1)).chunk(2, -1)
        logvar = logvar.clamp(-10, 6)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        pred = self.dec(torch.cat([f, z], -1))
        recon = (pred - target).abs().mean() if mask is None else mmean((pred - target).abs())
        if self.kind == "cvae":
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        else:
            pm, plv = self.prior(f).chunk(2, -1)
            plv = plv.clamp(-10, 6)
            kl = (0.5 * (plv - logvar + (logvar.exp() + (mu - pm).pow(2)) / plv.exp() - 1)).sum(-1).mean()
        with torch.no_grad():
            d["l1_policy"] = (self.act(f)[0] - target).abs().mean()
        d.update(recon=recon.detach(), kl=kl.detach())
        return recon + self.beta * kl, d

    def velocity(self, f, x, t):
        return self.vel(torch.cat([f, x, time_embed(t, self.tdim)], -1))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def act(self, f, mode="argmax"):
        """-> action (n,D), mode index (n,) or None, mode probs (n,K) or None.
        mode: 'argmax' (WTA/cat: most probable mode; cvae: z=0; cvae_lp: prior mean)
              'sample' (sample the mode / z from the (learned) prior; cvae: z ~ N(0,I))."""
        n = f.shape[0]
        if self.kind == "l1":
            return self.reg(f), None, None
        if self.kind == "fm":
            x = torch.zeros(n, self.D, device=f.device, dtype=f.dtype)
            if mode == "sample":
                x = torch.randn_like(x)
            N = int(self.fm_steps)
            for i in range(N):
                t = torch.full((n,), i / N, device=f.device, dtype=f.dtype)
                x = x + self.velocity(f, x, t) / N
            return x, None, None
        if self.discrete:
            logits = self.mode_logits(f)
            p = logits.softmax(-1)
            k = logits.argmax(-1) if mode == "argmax" else torch.multinomial(p, 1)[:, 0]
            preds = self.all_heads(f)
            return preds[torch.arange(n, device=f.device), k], k, p
        if self.kind == "cvae":
            z = torch.zeros(n, self.zdim, device=f.device, dtype=f.dtype)
            if mode == "sample":
                z = torch.randn_like(z)
        else:
            pm, plv = self.prior(f).chunk(2, -1)
            z = pm if mode == "argmax" else pm + (0.5 * plv.clamp(-10, 6)).exp() * torch.randn_like(pm)
        return self.dec(torch.cat([f, z], -1)), None, None

    @torch.no_grad()
    def mode_diagnostics(self, f, target):
        """Held-out classifier quality against the WTA winner (argmin-L1 head) labels."""
        if not self.discrete:
            return {}
        per = (self.all_heads(f) - target[:, None]).abs().mean(-1)
        win = per.argmin(1)
        p = self.mode_logits(f).softmax(-1)
        nll = F.nll_loss(p.clamp_min(1e-12).log(), win)
        base = torch.bincount(win, minlength=self.K).float().max() / len(win)
        return dict(acc=float((p.argmax(-1) == win).float().mean()), nll=float(nll), ece=ece(p, win),
                    base_rate=float(base), n=int(len(win)),
                    win_freq=(torch.bincount(win, minlength=self.K).float() / len(win)).tolist(),
                    l1_win=float(per.gather(1, win[:, None]).mean()),
                    l1_argmax=float(per.gather(1, p.argmax(-1)[:, None]).mean()))
