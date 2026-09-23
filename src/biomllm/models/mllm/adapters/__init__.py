from biomllm.models.mllm.adapters.toy import ToyMLLM

ADAPTERS = {
    "toy": ToyMLLM,
}


def build_mllm(name: str, **kwargs):
    if name not in ADAPTERS:
        raise KeyError(f"unknown MLLM adapter '{name}', available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)
