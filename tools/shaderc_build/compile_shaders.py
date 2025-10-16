#!/usr/bin/env python3
"""
Placeholder shader compiler wrapper.

Intended behavior:
- Scan the shaders/ directory for .comp/.glsl files.
- Invoke glslc or shaderc to produce SPIR-V into build/shaders/.
- Inject common includes via `shaders/common.glsl` and `tools/shaderc_build/shader_pch.glsl`.

This is a stub; implementation will be added later.
"""

import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    print(f"[shaderc_build] Placeholder run. Root: {root}")
    print("TODO: Implement GLSL→SPIR-V compilation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

