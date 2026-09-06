# Planned dependency and license review

Review date: 2026-09-06
Python floor: 3.11

This is an engineering inventory based on official project documentation, repository license
files, and current package-index metadata. It is not legal advice. Dependency licenses do not
select or imply a license for this repository's original code.

| Dependency | Intended purpose | Observed current release / Python metadata | License evidence | Distribution consideration | Decision / first phase |
|---|---|---|---|---|---|
| NumPy | Array contract and later image computation | 2.5.3 requires Python >=3.12; 2.3.5 supports >=3.11 | BSD-3-Clause core; current metadata also identifies bundled permissive components | Preserve package notices when redistributing binaries | Approved now as `>=1.26,<2.4` to retain Python 3.11 / Phase 4 |
| Pillow | Image decoding and later shaped rendering | 12.3.0 requires Python >=3.10 | MIT-CMU | Wheels contain native libraries with their own notices | Approved when implemented / Phase 5 |
| opencv-python-headless | Headless CV primitives | 5.0.0.93 publishes Python 3.11-compatible wheels | Packaging scripts MIT; OpenCV Apache-2.0 | Wheels include FFmpeg and other third-party binaries/licenses; use headless build | Approved with wheel notice review / Phase 5 |
| pypdfium2 | PDF rasterization | 5.13.0 metadata supports Python >=3.6; project provides broad prebuilt packages | Bindings Apache-2.0 OR BSD-3-Clause; PDFium BSD-style | PDFium and all bundled third-party license files must accompany binary redistribution as applicable | Approved with exact wheel inventory / Phase 5 |
| PyTorch | Sole ML framework | 2.14.0 requires Python >=3.10 and has CPython 3.11 wheels | BSD-style project license plus multiple bundled component licenses | Large platform-specific wheels; retain notices and do not bundle unnecessarily | Approved for CPU/GPU-specific installation guidance / Phase 9 |
| safetensors | Tensor-only model weights | 0.8.0 requires Python >=3.10 and supports CPython 3.11 | Apache-2.0 | Preserve license/notice when redistributing; validate format limits independently | Approved / Phase 10 |
| FastAPI | Thin typed HTTP adapter | 0.141.1 requires Python >=3.10 | MIT | Transitive Starlette/Pydantic inventory required when installed | Approved as optional dependency / Phase 13 |
| Uvicorn | Reference ASGI server | 0.52.4 requires Python >=3.10 | BSD-3-Clause | Optional `standard` extra adds native/transitive dependencies, so base package is preferred | Approved as optional dependency / Phase 13 |
| python-multipart | Multipart upload parsing | 0.0.32 requires Python >=3.10 | Apache-2.0 | Input limits remain application responsibilities; preserve notice if redistributed | Approved as optional dependency / Phase 13 |
| pytest | Test runner | 9.1.1 requires Python >=3.10 | MIT | Development-only | Approved now as `>=8.4,<10` / Phase 4 |
| pytest-cov | Coverage reporting | 7.1.0 requires Python >=3.9 | MIT | Development-only | Approved now as `>=6,<8` / Phase 4 |
| Ruff | Linter and formatter | 0.16.6 requires Python >=3.7 and publishes platform packages | MIT | Development-only Rust binary | Approved now as `>=0.12,<0.17` / Phase 4 |

## Authoritative references

- [NumPy license](https://numpy.org/doc/stable/license)
- [Pillow source and license](https://github.com/python-pillow/Pillow)
- [OpenCV Python packaging and bundled licenses](https://github.com/opencv/opencv-python)
- [pypdfium2 packaging and licensing](https://github.com/pypdfium2-team/pypdfium2)
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [PyTorch installation support](https://docs.pytorch.org/get-started/locally/)
- [safetensors source and license](https://github.com/huggingface/safetensors)
- [FastAPI source/documentation](https://fastapi.tiangolo.com/)
- [Uvicorn source](https://github.com/Kludex/uvicorn)
- [python-multipart source](https://github.com/Kludex/python-multipart)
- [pytest source and license](https://github.com/pytest-dev/pytest)
- [Ruff source and license](https://github.com/astral-sh/ruff)

Exact selected versions, transitive packages, wheel contents, hashes, and notices are reviewed in
the phase where each dependency first enters installation metadata. Runtime code must not
auto-download models, fonts, or datasets.

## CI action review

Phase 4 CI uses `actions/checkout@v6` and `actions/setup-python@v7`. The reviewed stable releases
were checkout 6.0.2 and setup-python 7.0.0; both current major lines use the maintained Node 24
runtime. The workflow grants only `contents: read` and explicitly selects Python 3.11. Immutable
commit-SHA pins remain a required pre-publication hardening step in Phase 14.
