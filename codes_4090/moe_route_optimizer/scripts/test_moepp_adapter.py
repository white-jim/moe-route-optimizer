import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from interfaces.hf_accelerate_adapter import create_hf_accelerate_adapter


def main():
    model_path = "../models/MoEPP-7B"
    adapter = create_hf_accelerate_adapter(
        model_path,
        ep_size=2,
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )
    try:
        print("loaded")
        print("num_moe_layers=", adapter._num_moe_layers)
        print(
            "first_gate=",
            type(adapter.get_first_moe_gate()).__name__
            if adapter.get_first_moe_gate() is not None
            else None,
        )
        print(
            "first_block=",
            type(adapter.get_first_moe_block()).__name__
            if adapter.get_first_moe_block() is not None
            else None,
        )
    finally:
        adapter.cleanup()
        print("cleanup")


if __name__ == "__main__":
    main()
