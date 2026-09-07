# Dependency and license review

Review date: 2026-09-07
Python floor: 3.11

This is an engineering inventory based on official project documentation, repository license
files, and current package-index metadata. It is not legal advice. Dependency licenses do not
select or imply a license for this repository's original code.

| Dependency | Intended purpose | Observed current release / Python metadata | License evidence | Distribution consideration | Decision / first phase |
|---|---|---|---|---|---|
| NumPy | Array contract and later image computation | 2.5.3 requires Python >=3.12; 2.3.5 supports >=3.11 | BSD-3-Clause core; current metadata also identifies bundled permissive components | Preserve package notices when redistributing binaries | Approved now as `>=1.26,<2.4` to retain Python 3.11 / Phase 4 |
| Pillow | Image decoding | 12.3.0 requires Python >=3.10 and installed on CPython 3.11 | MIT-CMU | Installed wheel records `dist-info/licenses/LICENSE`; preserve it if redistributing the wheel | Approved as `>=11,<13` / Phase 5 |
| opencv-python-headless | Headless CV primitives | 5.0.0.93 is current; selected 4.14.0.94 installed on CPython 3.11 and requires NumPy >=2 there | Packaging scripts MIT; installed OpenCV wheel metadata is Apache-2.0 | Wheel includes `LICENSE.txt`, `LICENSE-3RD-PARTY.txt`, and an FFmpeg binary; preserve both notice files | Approved as `>=4.10,<5` to avoid an unreviewed major transition / Phase 5 |
| pypdfium2 | PDF rasterization | 5.13.0 metadata supports Python >=3.6 and installed on CPython 3.11 | Bindings Apache-2.0 OR BSD-3-Clause; PDFium and binary-build components have their recorded licenses | Wheel contains PDFium plus Apache/BSD/CC-BY texts and per-component build licenses for FreeType, ICU, JPEG, OpenJPEG, PNG, TIFF, zlib, and others; preserve the complete wheel license inventory in binary redistribution | Approved as `>=5,<6` / Phase 5 |
| uharfbuzz | Contextual Urdu shaping | 0.56.1 on CPython 3.11, bundling HarfBuzz 14.4.0 | Apache-2.0 package metadata and license | Preserve its license when redistributing an environment; no runtime download | Approved as `>=0.56,<0.57` for direct shaping / Phase 8 |
| freetype-py | Glyph rasterization | 2.5.1 on CPython 3.11, reporting FreeType 2.13.2 | BSD-3-Clause package metadata; bundled FreeType has its upstream license | Preserve wrapper and bundled-library notices when redistributing an environment | Approved as `>=2.5,<3` / Phase 8 |
| Noto Nastaliq Urdu font | Reproducible Urdu fixture rendering | Exact `NotoNastaliqUrdu-v4.000` release asset; unmodified variable TTF, 325,092 bytes | SIL Open Font License 1.1 and upstream copyright stored beside binary | OFL permits bundling/redistribution with copyright and license; do not sell font alone; no reserved names recorded; distinct from project source | Approved bundled static asset / Phase 8 |
| PyTorch | Sole ML framework | Exact tested `2.14.0+cpu`; upstream 2.14.0 requires Python >=3.10 and publishes CPython 3.11 Windows and manylinux wheels | Project describes a BSD-style license; installed 2.14 metadata expresses Apache-2.0, Apache-2.0 with LLVM exception, BSD-2/3-Clause, BSL-1.0, and MIT components | Official Windows CPU wheel was 124.0 MB; its `dist-info/licenses/` contains the project and third-party license inventory, which must accompany binary redistribution | Approved as optional `torch>=2.14,<2.15`; official CPU wheel channel used for reference/CI / Phase 9 |
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
- [Pillow image and decompression-bomb documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [Pillow EXIF transpose documentation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html)
- [OpenCV Python packaging and bundled licenses](https://github.com/opencv/opencv-python)
- [pypdfium2 packaging and licensing](https://github.com/pypdfium2-team/pypdfium2)
- [pypdfium2 Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
- [uharfbuzz source and license](https://github.com/harfbuzz/uharfbuzz)
- [freetype-py source and license](https://github.com/rougier/freetype-py)
- [Noto Nastaliq Urdu v4.000 release](https://github.com/notofonts/nastaliq/releases/tag/NotoNastaliqUrdu-v4.000)
- [SIL Open Font License 1.1](https://openfontlicense.org/)
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [PyTorch installation support](https://docs.pytorch.org/get-started/locally/)
- [PyTorch 2.14.0 package metadata and CPython 3.11 wheels](https://pypi.org/project/torch/2.14.0/)
- [safetensors source and license](https://github.com/huggingface/safetensors)
- [FastAPI source/documentation](https://fastapi.tiangolo.com/)
- [Uvicorn source](https://github.com/Kludex/uvicorn)
- [python-multipart source](https://github.com/Kludex/python-multipart)
- [pytest source and license](https://github.com/pytest-dev/pytest)
- [Ruff source and license](https://github.com/astral-sh/ruff)

The Phase 5 clean Python 3.11 resolution was NumPy 2.3.5, Pillow 12.3.0,
opencv-python-headless 4.14.0.94, and pypdfium2 5.13.0. The three new dependencies introduced no
additional Python-package transitive dependency beyond OpenCV's existing NumPy requirement.
Runtime code must not auto-download models, fonts, or datasets.

Phase 7 adds no runtime or development dependency. Manifest, fingerprint, Unicode, split, and
review behavior use the Python standard library; line-image validation reuses the already approved
Pillow dependency.

Phase 8 adds only the two small shaping/rasterization wrappers above. Pillow 12.3.0 on the
reference Windows CPython 3.11 environment reports FreeType 2.14.3 but no RAQM; the direct
uharfbuzz 0.56.1/HarfBuzz 14.4.0 plus freetype-py 2.5.1/FreeType 2.13.2 route is therefore used and
tested instead of relying on Pillow layout. No FriBiDi package is added: controlled single-script
fixture lines use explicit RTL with isolated numeric LTR runs, and the limitation for arbitrary
mixed-language bidi text is documented. `fonttools` is unnecessary. Phase 8 added no model
framework.

Phase 9 adds PyTorch only under the independently installable `ml` extra, bounded to the reviewed
2.14 minor line. The reference Windows CPython 3.11 environment resolved `torch 2.14.0+cpu` from
`https://download.pytorch.org/whl/cpu`; `torch.version.cuda` was `None` and the CPU backend was
available. The upstream 2.14.0 metadata requires Python >=3.10 and publishes CPython 3.11 wheels
for Windows x86-64 and Linux x86-64/aarch64. The CPU wheel records a project license plus a broad
third-party license tree; no part of that inventory is copied into this source repository, but the
wheel notices must remain intact in redistributed environments. PyTorch's pure-Python transitive
resolution here included filelock, fsspec, Jinja2/MarkupSafe, networkx, SymPy/mpmath,
typing-extensions, and setuptools. Runtime code performs no package, model, or weight download.
No `safetensors` dependency is installed before Phase 10 needs checkpoint persistence.

## CI action review

CI uses `actions/checkout@v6` and `actions/setup-python@v7`. The reviewed stable releases
were checkout 6.0.2 and setup-python 7.0.0; both current major lines use the maintained Node 24
runtime. The workflow grants only `contents: read` and explicitly selects Python 3.11. The core
job installs no ML extra and exercises all non-ML tests plus optional-import isolation. The ML CPU
job installs the exact official 2.14.0 CPU wheel, installs `[ml,dev]`, runs the full suite with an
85% coverage gate, and runs `pip check`. Immutable commit-SHA pins remain a required
pre-publication hardening step in Phase 14.
