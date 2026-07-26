"""Smoke test for the dedicated offset feature path in JointPolicyObjective.

Stubs the tokenizer / episode / decoders; exercises _context / _features /
_offset_features / _predict / compute_metrics and checks:
  - offset_decoder=None  -> offset uses the shared code feature (back-compat)
  - offset_decoder set   -> offset uses its own pooler (different from code path)
  - compute_metrics runs and returns code_0..3 + offset losses
  - with encoder/decoder/code_head frozen, offset-loss grad reaches ONLY the
    offset branch (offset_decoder + offset_head), not decoder/code_head
"""
import torch
from torch import Tensor, nn

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict as RModuleDict
from rmind.components.objectives.joint_policy import JointPolicyObjective

B, D = 3, 384
G, C, AH, ASP = 4, 16, 6, 4
ADIM = AH * ASP  # 24


class Pooler(nn.Module):
    """Stand-in cross-attn head: mean-pool context, project, keep query dim."""
    def __init__(self, tag):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.tag = tag

    def forward(self, inp):
        ctx = inp["context"]  # (b, n, d)
        return self.lin(ctx.mean(-2, keepdim=True))  # (b, 1, d)


class Tok(nn.Module):
    class Q:
        num_quantizers, codebook_size = G, C
    quantizer = Q()
    _action_features = ASP

    def __call__(self, chunk):  # (b, AH, ASP) -> codes (b, G)
        b = chunk.shape[0]
        return torch.randint(0, C, (b, G))

    def invert(self, codes):  # (b, G) -> (b, ADIM)
        return codes.float().sum(-1, keepdim=True).expand(-1, ADIM) * 0.01

    def _normalize(self, x):  # (b, ADIM) -> (b, ADIM)
        return x


class Parsed:
    def __init__(self, t): self.t = t
    def get(self, _k): return self.t


class Sel:
    def __init__(self, t): self.t = t
    def parse(self, _e): return Parsed(self.t)


_OS = torch.randn(B, 64, D)  # fixed observation_summary tokens
_OH = torch.randn(B, 32, D)  # fixed observation_history tokens
_MASK = torch.randn(B, 2, 8, D)  # fixed utility mask embeddings
_CHUNK = torch.randn(B, 2, AH, ASP)  # (b, t, action_clip, action_space); [:, -1] -> (b, AH, ASP)


class Last:
    def select(self, k):
        return Sel(_OS if k[1] == SummaryToken.OBSERVATION_SUMMARY else _OH)


class Emb:
    def get(self, _k): return _MASK  # (b, t, nmask, d)


class Ep:
    index = [None, Last()]
    embeddings = Emb()
    def get(self, _chunk): return _CHUNK


def make(offset_decoder):
    losses = RModuleDict(modules={"code": nn.CrossEntropyLoss(), "offset": nn.L1Loss()})
    return JointPolicyObjective(
        tokenizer=Tok(),
        decoder=Pooler("code"),
        code_head=nn.Linear(D, G * C),
        offset_head=nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, G * C * ADIM)),
        losses=losses,
        chunk=("input", "joint_actions"),
        norm=nn.LayerNorm(D),
        sample_codes=False,
        offset_decoder=offset_decoder,
    )


emb = torch.randn(B, 8, D)
ep = Ep()

# back-compat: None -> offset feature == code feature
obj0 = make(None)
assert torch.equal(obj0._offset_features(ep, emb), obj0._features(ep, emb))
print("[back-compat] offset_decoder=None -> offset uses shared code feature: OK")

# dedicated path: offset feature comes from offset_decoder, differs from code path
obj = make(Pooler("offset"))
cf, of = obj._features(ep, emb), obj._offset_features(ep, emb)
assert cf.shape == of.shape == (B, D) and not torch.allclose(cf, of)
print(f"[dedicated] code vs offset features differ, both {tuple(of.shape)}: OK")

# compute_metrics end-to-end
m = obj.compute_metrics(episode=ep, embedding=emb)
losses = m["loss"]
assert set(losses) == {"code_0", "code_1", "code_2", "code_3", "offset"}, set(losses)
print(f"[compute_metrics] losses={ {k: round(float(v),3) for k,v in losses.items()} }")

# gradient isolation: freeze encoder-side (decoder + code_head + norm), backward,
# check grad reaches offset branch only
for mod in (obj.decoder, obj.code_head, obj.norm):
    for p in mod.parameters():
        p.requires_grad_(False)
obj.zero_grad()
losses["offset"].backward()
off_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.offset_decoder.parameters())
oh_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.offset_head.parameters())
code_grad = any(p.grad is not None for p in obj.code_head.parameters())
dec_grad = any(p.grad is not None for p in obj.decoder.parameters())
assert off_grad and oh_grad, (off_grad, oh_grad)
assert not code_grad and not dec_grad, (code_grad, dec_grad)
print("[freeze] offset-loss grad reaches offset_decoder+offset_head only (code/decoder frozen): OK")

print("\nALL SMOKE CHECKS PASSED")
