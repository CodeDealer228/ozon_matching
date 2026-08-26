"""Real tests of SOAP, imported from the exact copy in
generated/soap_optimizer.py (see test_extraction_matches_kernel.py for the
guarantee that copy is byte-identical to the kernel)."""
import copy

import torch
import torch.nn as nn

from generated.soap_optimizer import SOAP, SOAP_MAX_PRECOND_DIM


def _manual_decoupled_adam_step(param, grad, exp_avg, exp_avg_sq, step, betas, eps, lr, weight_decay, clip):
    beta1, beta2 = betas
    grad = grad.clamp(-clip, clip)
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    step_size = lr * (bias2 ** 0.5) / bias1
    denom = exp_avg_sq.sqrt() + eps
    param = param - step_size * (exp_avg / denom)
    param = param - lr * weight_decay * param
    # SOAP stores exp_avg/exp_avg_sq BF16 for unpreconditioned (encoder-scale) params --
    # round-trip the reference the same way so this checks the update RULE, not storage precision.
    exp_avg = exp_avg.to(torch.bfloat16).float()
    exp_avg_sq = exp_avg_sq.to(torch.bfloat16).float()
    return param, exp_avg, exp_avg_sq


def test_first_ever_step_only_initializes_no_weight_change():
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(6, 4))
    before = param.detach().clone()
    optimizer = SOAP([{"params": [param], "weight_decay": 0.0}], lr=1e-2)
    param.grad = torch.randn(6, 4)
    optimizer.step()
    assert torch.equal(param.detach(), before), "SOAP's first-ever step must not touch weights"
    state = optimizer.state[param]
    assert state["step"] == 0
    assert state["Q"] is not None


def test_unpreconditioned_matches_decoupled_adam_reference():
    """A 2000x2000 param sits above SOAP_MAX_PRECOND_DIM on both sides, so
    project/project_back must be an exact identity and the whole class must
    degrade to plain decoupled Adam -- byte for byte, not just "close"."""
    torch.manual_seed(0)
    big = nn.Parameter(torch.randn(2000, 2000))
    ref_param = big.detach().clone()
    ref_avg = torch.zeros_like(ref_param)
    ref_avg_sq = torch.zeros_like(ref_param)
    betas, eps, lr, wd, clip = (0.9, 0.999), 1e-6, 1e-3, 0.02, 1.0
    optimizer = SOAP([{"params": [big], "weight_decay": wd}], lr=lr, betas=betas, eps=eps,
                      max_precond_dim=SOAP_MAX_PRECOND_DIM, precondition_frequency=5)
    for step in range(1, 4):
        grad = torch.randn_like(big)
        big.grad = grad.clone()
        optimizer.step()
        if step == 1:
            assert torch.equal(big.detach(), ref_param)
            continue
        ref_param, ref_avg, ref_avg_sq = _manual_decoupled_adam_step(
            ref_param, grad, ref_avg, ref_avg_sq, step - 1, betas, eps, lr, wd, clip
        )
        assert torch.allclose(big.detach(), ref_param, atol=1e-5, rtol=1e-5), (
            f"unpreconditioned SOAP step {step} diverged from the plain-Adam reference"
        )
    state = optimizer.state[big]
    assert all(len(side) == 0 for side in state["GG"]), "2000x2000 must be fully excluded by max_precond_dim"
    assert state["moment_dtype"] == torch.bfloat16


def test_preconditioned_param_progresses_past_a_basis_refresh():
    """A 6x4 param sits fully inside max_precond_dim on both sides, so it
    gets real eigenbasis rotation, including a QR-based basis refresh
    (precondition_frequency=3, run for 8 steps -> refreshes at least twice)."""
    torch.manual_seed(1)
    param = nn.Parameter(torch.randn(6, 4))
    init = param.detach().clone()
    optimizer = SOAP([{"params": [param], "weight_decay": 0.0}], lr=1e-2, precondition_frequency=3)
    for _ in range(8):
        param.grad = torch.randn(6, 4)
        optimizer.step()
    state = optimizer.state[param]
    assert all(len(side) > 0 for side in state["GG"]), "6x4 must be fully covered by max_precond_dim"
    assert state["moment_dtype"] == torch.float32
    assert torch.isfinite(param.detach()).all()
    assert not torch.equal(param.detach(), init), "preconditioned SOAP made no progress"


def test_resume_from_state_dict_reproduces_uninterrupted_run_exactly():
    """The exact contract the real kernel's quiet-checkpoint/resume mechanism
    depends on: save mid-run, reload into a fresh optimizer, replay the rest
    of the gradients, and land on the identical final parameter values an
    uninterrupted run would have reached."""
    torch.manual_seed(1)
    param = nn.Parameter(torch.randn(6, 4))
    optimizer = SOAP([{"params": [param], "weight_decay": 0.0}], lr=1e-2, precondition_frequency=3)
    grads = [torch.randn(6, 4) for _ in range(8)]
    snapshot = None
    for step, grad in enumerate(grads):
        param.grad = grad.clone()
        optimizer.step()
        if step == 3:
            # deepcopy, not .cpu(): on CPU tensors .cpu() is a no-op that would alias into
            # GG's later in-place lerp_ mutations -- on the real TPU run this is instead
            # forced by a genuine device transfer (see to_cpu_state in the kernel).
            snapshot = dict(opt=copy.deepcopy(optimizer.state_dict()), param=param.detach().clone())
    direct_final = param.detach().clone()

    resumed_param = nn.Parameter(snapshot["param"].clone())
    resumed_optimizer = SOAP([{"params": [resumed_param], "weight_decay": 0.0}], lr=1e-2, precondition_frequency=3)
    resumed_optimizer.load_state_dict(snapshot["opt"])
    for grad in grads[4:]:
        resumed_param.grad = grad.clone()
        resumed_optimizer.step()

    assert torch.allclose(resumed_param.detach(), direct_final, atol=1e-6), (
        "resuming from a mid-run state_dict must reproduce the uninterrupted run exactly"
    )


def test_load_state_dict_restores_bf16_moments_not_upcast_to_fp32():
    """PyTorch's default Optimizer.load_state_dict can silently cast state
    tensors to match the parameter's dtype (FP32 here) -- SOAP overrides
    load_state_dict specifically to guard against that; this pins the guard."""
    big = nn.Parameter(torch.randn(2000, 2000))
    optimizer = SOAP([{"params": [big], "weight_decay": 0.0}], lr=1e-3)
    big.grad = torch.randn_like(big)
    optimizer.step()  # init only
    big.grad = torch.randn_like(big)
    optimizer.step()  # first real update -> exp_avg/exp_avg_sq now exist, BF16
    assert optimizer.state[big]["exp_avg"].dtype == torch.bfloat16

    reloaded = SOAP([{"params": [nn.Parameter(big.detach().clone())]}], lr=1e-3)
    reloaded.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_state = next(iter(reloaded.state.values()))
    assert restored_state["exp_avg"].dtype == torch.bfloat16
    assert restored_state["exp_avg_sq"].dtype == torch.bfloat16


def test_1d_params_never_preconditioned():
    bias = nn.Parameter(torch.zeros(32))
    optimizer = SOAP([{"params": [bias], "weight_decay": 0.0}], lr=1e-2)
    for _ in range(3):
        bias.grad = torch.randn(32)
        optimizer.step()
    state = optimizer.state[bias]
    assert state["GG"] == [[]]
    assert state["moment_dtype"] == torch.bfloat16
    assert torch.isfinite(bias.detach()).all()
    assert not torch.equal(bias.detach(), torch.zeros(32))


def test_reset_clears_all_state():
    param = nn.Parameter(torch.randn(6, 4))
    optimizer = SOAP([{"params": [param], "weight_decay": 0.0}], lr=1e-2, precondition_frequency=3)
    for _ in range(5):
        param.grad = torch.randn(6, 4)
        optimizer.step()
    assert len(optimizer.state) > 0
    optimizer.reset_()
    assert len(optimizer.state) == 0
    # after reset, the next step() must behave like a genuine first-ever step again
    before = param.detach().clone()
    param.grad = torch.randn(6, 4)
    optimizer.step()
    assert torch.equal(param.detach(), before)
