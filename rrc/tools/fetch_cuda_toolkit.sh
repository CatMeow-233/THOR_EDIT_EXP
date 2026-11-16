#!/usr/bin/env bash
# rrc/tools/fetch_cuda_toolkit.sh — assemble a COMPILE-ONLY CUDA 12.9 toolkit (nvcc, headers,
# libcudart, libcublasLt) for `make -C rrc cuda-objs` / `make -C rrc cuda` on a machine without the
# NVIDIA installer, e.g. CI. Nothing here is needed for the CPU build, and no test runs it.
#
#   rrc/tools/fetch_cuda_toolkit.sh DEST            # download, verify sha256, assemble DEST/tk
#   rrc/tools/fetch_cuda_toolkit.sh DEST --from DIR # same, from already downloaded package files in DIR
#   make -C rrc cuda CUDA_HOME=DEST/tk
#
# Sources (pinned versions, every file checked against the sha256 below):
#   PyPI wheels (pip download):  nvidia-cuda-runtime-cu12 12.9.79, nvidia-cublas-cu12 12.9.3.13,
#     nvidia-cuda-cccl-cu12 12.9.27, nvidia-cuda-nvcc-cu12 12.9.86 (headers, ptxas, libdevice; the
#     Linux wheel has no nvcc driver, cicc or cudafe++)
#   conda packages from https://conda.anaconda.org/nvidia/linux-64/: cuda-nvcc-tools, cuda-crt-tools,
#     cuda-nvvm-tools, cuda-nvvm-impl 12.9.86 (nvcc, cudafe++, fatbinary, nvlink, cicc, libnvvm)
# developer.download.nvidia.com (the runfile / apt repositories) may be blocked by proxies; this
# script does not use it. The result can compile and link but contains no driver (libcuda): running
# anything needs a real GPU machine with an NVIDIA driver >= the 12.9 requirement.
# Requirements: bash, python3 with pip, and either the python `zstandard` module or `tar --zstd`.
set -euo pipefail

usage() {
    echo "usage: $0 DEST [--from DIR]" >&2
    exit 2
}
[ $# -ge 1 ] || usage
DEST=$1
shift
FROM=""
while [ $# -gt 0 ]; do
    case "$1" in
    --from) [ $# -ge 2 ] || usage; FROM=$2; shift 2 ;;
    *) usage ;;
    esac
done

WHEELS=(
    "nvidia-cuda-runtime-cu12==12.9.79 nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3"
    "nvidia-cublas-cu12==12.9.3.13 nvidia_cublas_cu12-12.9.3.13-py3-none-manylinux_2_27_x86_64.whl d7d51055f161b317c7105ddaae90e71eda059026074ffef93cf7cde2886bfe72"
    "nvidia-cuda-cccl-cu12==12.9.27 nvidia_cuda_cccl_cu12-12.9.27-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 37869e17ce2e1ecec6eddf1927cca0f8c34e64fd848d40453df559091e2d7117"
    "nvidia-cuda-nvcc-cu12==12.9.86 nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 5d6a0d32fdc7ea39917c20065614ae93add6f577d840233237ff08e9a38f58f0"
)
CONDA_URL=https://conda.anaconda.org/nvidia/linux-64
CONDAS=(
    "cuda-crt-tools-12.9.86-0.conda c187f0482242f5927c43a4c06c09fce05e3367f648d2914081e1e8333eda06f0"
    "cuda-nvcc-tools-12.9.86-0.conda 34247d80e740093e0e4a251423b051b5722b37a487b607002a705234d802ddc3"
    "cuda-nvvm-tools-12.9.86-0.conda 199e68950623c30c419eedf4dba2f88a245365e1e0e7602d71ee77a58864d06e"
    "cuda-nvvm-impl-12.9.86-0.conda 77277cdd8b8cb70ea695771749f3dee928ea67c1e57e2e072cb2251c9be72e6f"
)

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
DL="$DEST/downloads"
UNPACK="$DEST/unpacked"
TK="$DEST/tk"
mkdir -p "$DL"

check_sha() { # file sha256
    local got
    got=$(python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1")
    if [ "$got" != "$2" ]; then
        echo "sha256 mismatch for $1: got $got, expected $2" >&2
        exit 1
    fi
}

fetch_wheel() { # spec file sha
    local spec=$1 file=$2 sha=$3
    if [ -n "$FROM" ] && [ -f "$FROM/$file" ]; then
        cp "$FROM/$file" "$DL/$file"
    elif [ ! -f "$DL/$file" ]; then
        python3 -m pip download --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 \
            --platform manylinux2014_x86_64 --platform manylinux2010_x86_64 \
            --python-version 3 --implementation py --dest "$DL" "$spec"
    fi
    check_sha "$DL/$file" "$sha"
}

fetch_conda() { # file sha
    local file=$1 sha=$2
    if [ -n "$FROM" ] && [ -f "$FROM/$file" ]; then
        cp "$FROM/$file" "$DL/$file"
    elif [ ! -f "$DL/$file" ]; then
        python3 -I -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
            "$CONDA_URL/$file" "$DL/$file.part"
        mv "$DL/$file.part" "$DL/$file"
    fi
    check_sha "$DL/$file" "$sha"
}

for w in "${WHEELS[@]}"; do
    # shellcheck disable=SC2086
    fetch_wheel $w
done
for c in "${CONDAS[@]}"; do
    # shellcheck disable=SC2086
    fetch_conda $c
done

rm -rf "$UNPACK" "$TK"
mkdir -p "$UNPACK/wheels" "$UNPACK/conda" "$TK/bin" "$TK/include" "$TK/lib64" "$TK/nvvm"

# Extraction runs in isolated mode (-I) with the archives as data; member paths are checked.
python3 -I - "$DL" "$UNPACK" <<'PY'
import io, os, subprocess, sys, tarfile, zipfile
dl, out = sys.argv[1], sys.argv[2]
def safe_extract_zip(z, dest):
    for m in z.namelist():
        p = os.path.realpath(os.path.join(dest, m))
        if not p.startswith(os.path.realpath(dest) + os.sep):
            raise SystemExit("unsafe path in archive: " + m)
    z.extractall(dest)
def zstd_decompress(data):
    try:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    except ImportError:
        return subprocess.run(["zstd", "-dc"], input=data, stdout=subprocess.PIPE, check=True).stdout
for f in sorted(os.listdir(dl)):
    path = os.path.join(dl, f)
    if f.endswith(".whl"):
        with zipfile.ZipFile(path) as z:
            safe_extract_zip(z, os.path.join(out, "wheels"))
    elif f.endswith(".conda"):
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():
                if m.startswith("pkg-") and m.endswith(".tar.zst"):
                    data = zstd_decompress(z.read(m))
                    with tarfile.open(fileobj=io.BytesIO(data)) as t:
                        dest = os.path.join(out, "conda")
                        for mem in t.getmembers():
                            p = os.path.realpath(os.path.join(dest, mem.name))
                            if not p.startswith(os.path.realpath(dest) + os.sep):
                                raise SystemExit("unsafe path in archive: " + mem.name)
                            if mem.issym() or mem.islnk():
                                tgt = os.path.realpath(os.path.join(os.path.dirname(p), mem.linkname))
                                if not tgt.startswith(os.path.realpath(dest) + os.sep):
                                    raise SystemExit("unsafe link in archive: " + mem.name)
                        t.extractall(dest)
PY

NV="$UNPACK/wheels/nvidia"
for d in cuda_runtime cublas cuda_cccl cuda_nvcc; do
    cp -a "$NV/$d/include/." "$TK/include/"
done
cp -a "$NV/cuda_runtime/lib/." "$NV/cublas/lib/." "$TK/lib64/"
cp -a "$NV/cuda_nvcc/nvvm/." "$TK/nvvm/"
# nvcc driver, front ends and cicc from the conda packages (they override the wheel's ptxas/nvvm)
cp -a "$UNPACK/conda/bin/." "$TK/bin/"
cp -a "$UNPACK/conda/nvvm/." "$TK/nvvm/"
rm -f "$TK/include/__init__.py" "$TK/lib64/__init__.py"
(cd "$TK/lib64" && ln -sf libcudart.so.12 libcudart.so && ln -sf libcublas.so.12 libcublas.so &&
    ln -sf libcublasLt.so.12 libcublasLt.so)

"$TK/bin/nvcc" --version
echo "compile-only CUDA toolkit ready: make -C rrc cuda-objs CUDA_HOME=$TK"
