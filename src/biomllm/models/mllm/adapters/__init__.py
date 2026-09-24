from biomllm.models.mllm.adapters.toy import ToyMLLM


def _lazy_qwen_vl(**kwargs):
    from biomllm.models.mllm.adapters.qwen_vl import QwenVLAdapter
    return QwenVLAdapter(**kwargs)


ADAPTERS = {
    "toy": ToyMLLM,
    "qwen_vl": _lazy_qwen_vl,
}


def build_mllm(name: str, **kwargs):
    if name not in ADAPTERS:
        raise KeyError(f"unknown MLLM adapter '{name}', available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)
