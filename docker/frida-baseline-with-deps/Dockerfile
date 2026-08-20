# Custom image candidate: judging baseline + FRIDA's runtime deps pre-installed.
#
# WHY: every FRIDA submission currently ships ~26 wheels in the zip and runs
# maybe_offline_install() (submits/frida/*/src/utils.py) at container start --
# a real pip subprocess (--no-index --no-deps) unpacking every wheel into a
# /tmp-style target dir, on EVERY fresh container. Baking these into the image
# instead removes that step from the critical path entirely: `import peft`
# just works, no subprocess, no wheel unzip, no offline-install code path at
# all needed in src/utils.py.
#
# Versions pinned to exactly what's already validated in
# submits/frida/tpu_human_256_epoch4/wheels/ -- same packages this project's
# own offline-install already trusts, not a fresh, unverified resolve.
#
# NOT a fix for Public/Private-stage timing on its own -- this only removes
# FIXED per-container startup overhead (pip install + wheel unzip), which is
# paid once per container lifetime, not per pair. It does not touch GPU
# compute time, which scales with PAIR_MAX_LENGTH x pair count and is the
# newer, larger bottleneck at Public scale (see README.md's "Check пройден,
# но Public — ещё нет"). Expected effect: safety margin back at the Check
# stage (plausibly enough to run PAIR_MAX_LENGTH=256 there again), not a
# Public/Private fix by itself.

FROM odsai/ecup26-matching-baseline:1.0

RUN pip install --no-cache-dir --no-deps \
    peft==0.20.0 \
    accelerate==1.14.0 \
    huggingface_hub==1.27.0 \
    click==8.4.2 \
    hf_xet==1.6.0 \
    httpx==0.28.1 \
    httpcore==1.0.9 \
    h11==0.16.0 \
    anyio==4.14.2 \
    idna==3.18 \
    certifi==2026.7.22 \
    colorama==0.4.6 \
    typer==0.27.1 \
    annotated-doc==0.0.5 \
    rich==15.0.0 \
    pygments==2.20.0 \
    markdown-it-py==4.2.0 \
    mdurl==0.1.2 \
    shellingham==1.5.4 \
    packaging==26.3 \
    tqdm==4.70.0 \
    filelock==3.32.3 \
    fsspec==2026.7.0 \
    typing_extensions==4.16.0 \
    psutil==7.2.2 \
    pyyaml==6.0.3 \
    safetensors==0.8.0

# Sanity check baked in at build time, not left to the judging run to
# discover: import peft NOW, not for the first time on the real stand.
RUN python -c "import peft, accelerate, huggingface_hub, safetensors; print('deps OK')"
