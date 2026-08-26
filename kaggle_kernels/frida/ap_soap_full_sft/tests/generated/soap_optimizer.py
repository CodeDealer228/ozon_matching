"""Exact copy of SOAP, extracted from frida_ap_soap_full_sft.py.
Regenerate with generate_extracted_sources.py -- do not hand-edit.
"""
import torch
from torch.optim import Optimizer


SOAP_BETAS = (0.9, 0.999)
SOAP_EPS = 1e-6
WEIGHT_DECAY = 0.02
SOAP_SHAMPOO_BETA = 0.999
SOAP_PRECONDITION_FREQUENCY = 10
SOAP_MAX_PRECOND_DIM = 1024
SOAP_CLIP_VALUE = 1.0


class SOAP(Optimizer):
    """SOAP (Vyas et al., arxiv:2409.11321): Adam run in Shampoo's eigenbasis.

    Close port of the reference implementation (github.com/nikhilvyas/SOAP),
    trimmed of conv-only machinery this model has no use for (merge_dims,
    channels_last, normalize_grads, precondition_1d -- 1D params here are
    always small biases/LayerNorms, never worth preconditioning), plus this
    project's FP32-master memory contract: exp_avg/exp_avg_sq are stored
    BF16 for any parameter SOAP_MAX_PRECOND_DIM leaves fully unpreconditioned
    (the encoder body -- huge tensors, and project/project_back are an exact
    identity there, so BF16 costs nothing beyond the rounding a plain BF16
    Adam moment already implies), and FP32 for any parameter that gets real
    eigenbasis rotation (the head -- a few MB, so full precision is free and
    avoids compounding round-trip error in a rotating basis). Weight decay
    is decoupled (AdamW-style, applied to the parameter directly), matching
    the reference -- NOT coupled into the gradient like a from-scratch Adam.
    """

    def __init__(self, params, lr, betas=SOAP_BETAS, eps=SOAP_EPS, weight_decay=WEIGHT_DECAY,
                 shampoo_beta=SOAP_SHAMPOO_BETA, precondition_frequency=SOAP_PRECONDITION_FREQUENCY,
                 max_precond_dim=SOAP_MAX_PRECOND_DIM, clip=SOAP_CLIP_VALUE):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         shampoo_beta=shampoo_beta, precondition_frequency=precondition_frequency,
                         max_precond_dim=max_precond_dim, clip=clip)
        super().__init__(params, defaults)

    @staticmethod
    def _covariance_side(size, max_precond_dim, device):
        if size > max_precond_dim:
            return []
        return torch.zeros(size, size, dtype=torch.float32, device=device)

    @classmethod
    def _init_state(cls, grad, state, group):
        state["step"] = 0
        if grad.dim() > 1:
            state["GG"] = [cls._covariance_side(side, group["max_precond_dim"], grad.device)
                           for side in grad.shape]
        else:
            state["GG"] = [[]]  # 1D params (biases, LayerNorm) are never preconditioned
        state["Q"] = None
        moment_dtype = torch.float32 if any(len(side) for side in state["GG"]) else torch.bfloat16
        state["moment_dtype"] = moment_dtype
        state["exp_avg"] = torch.zeros_like(grad, dtype=moment_dtype)
        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=moment_dtype)

    @staticmethod
    def _orthogonal_basis(covariances):
        basis = []
        for covariance in covariances:
            if not len(covariance):
                basis.append([])
                continue
            eye = torch.eye(covariance.shape[0], dtype=torch.float32, device=covariance.device)
            _, vectors = torch.linalg.eigh(covariance + 1e-30 * eye)
            basis.append(torch.flip(vectors, [1]))
        return basis

    @staticmethod
    def _refresh_basis(covariances, basis):
        # The reference implementation also sorts columns by estimated eigenvalue before the
        # QR step (torch.argsort) and permutes exp_avg_sq to match, to keep "coordinate i"
        # meaning roughly the same thing across refreshes. Dropped here: a real TPU run hit a
        # fatal XLA-internal crash inside this exact preconditioner-refresh window (`Check
        # failed: state Expected an array shape. Got (f32[32], s64[32])` in shape.cc -- the
        # (values, indices) tuple shape torch.argsort/sort produce, at the first
        # precondition_frequency boundary of real training) -- consistent with argsort not
        # being reliably lowered by this torch_xla/TPU build. The power iteration + QR still
        # converges to a valid orthonormal basis for the SAME subspace without the sort; only
        # which physical column ends up "coordinate 0" becomes unordered across refreshes, so
        # exp_avg_sq's per-coordinate estimate is very occasionally attributed to a shuffled
        # coordinate right after a refresh -- a minor, EMA-smoothed cost, not a correctness bug.
        refreshed = []
        for covariance, vectors in zip(covariances, basis):
            if not len(covariance):
                refreshed.append([])
                continue
            power_iterate = covariance @ vectors
            new_vectors, _ = torch.linalg.qr(power_iterate)
            refreshed.append(new_vectors)
        return refreshed

    @staticmethod
    def _rotate(value, basis, transpose):
        """transpose=False projects raw -> eigenbasis; transpose=True is the inverse."""
        for vectors in basis:
            if len(vectors):
                value = torch.tensordot(value, vectors, dims=[[0], [1 if transpose else 0]])
            else:
                value = value.permute(*range(1, value.dim()), 0)
        return value

    @staticmethod
    def _update_covariance(grad, state, group):
        for index, side in enumerate(state["GG"]):
            if not len(side):
                continue
            contracted = [dimension for dimension in range(grad.dim()) if dimension != index]
            outer_product = torch.tensordot(grad, grad, dims=[contracted, contracted])
            side.lerp_(outer_product, 1.0 - group["shampoo_beta"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float().clamp(-group["clip"], group["clip"])
                state = self.state[parameter]

                if "step" not in state:
                    self._init_state(grad, state, group)
                    self._update_covariance(grad, state, group)
                    state["Q"] = self._orthogonal_basis(state["GG"])
                    continue  # first-ever step only initializes GG/Q, matching the reference --
                              # it must never precondition using the gradient that built Q.

                old_basis = state["Q"]
                grad_projected = self._rotate(grad, old_basis, transpose=False)
                exp_avg = state["exp_avg"].float().mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq = state["exp_avg_sq"].float().mul_(beta2).addcmul_(
                    grad_projected, grad_projected, value=1.0 - beta2
                )
                state["step"] += 1
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] * (bias2 ** 0.5) / bias1
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                update = self._rotate(exp_avg / denominator, old_basis, transpose=True)
                parameter.add_(update, alpha=-step_size)
                if group["weight_decay"] > 0.0:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])

                # Preconditioner refresh, exactly like the reference's update_preconditioner:
                # un-rotate the just-updated running mean out of the OLD basis, fold this
                # step's raw gradient into GG, maybe refresh the basis (a cheap power-iteration
                # + QR, not a full eigh -- that only ever runs once, at init, above), then
                # re-rotate exp_avg for next step's use.
                raw_exp_avg = self._rotate(exp_avg, old_basis, transpose=True)
                self._update_covariance(grad, state, group)
                new_basis = old_basis
                if state["step"] % group["precondition_frequency"] == 0:
                    new_basis = self._refresh_basis(state["GG"], old_basis)
                state["Q"] = new_basis
                state["exp_avg"] = self._rotate(raw_exp_avg, new_basis, transpose=False).to(state["moment_dtype"])
                state["exp_avg_sq"] = exp_avg_sq.to(state["moment_dtype"])
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            moment_dtype = state.get("moment_dtype", torch.float32)
            state["exp_avg"] = state["exp_avg"].to(moment_dtype)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(moment_dtype)

    @torch.no_grad()
    def reset_(self):
        self.state.clear()
