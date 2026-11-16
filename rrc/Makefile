# rrc — build. All outputs go to $(BUILD) (default rrc/build/, gitignored).
#
#   make cpu                 static library + binaries with the CPU backend
#   make test                build and run the C unit tests (+ nvcc header check if CUDA is set)
#   make cuda CUDA_HOME=...  build librrc_cuda.a + rrc_selftest with nvcc (CUDA_ARCH or ARCH, default sm_120)
#   make cuda-objs CUDA_HOME=...  compile every cuda/*.cu to an object only (CI, no GPU, no link)
#   make asan                build and run the C tests (+ rrc_selftest_host) with -fsanitize=address,undefined
#   make clean
#
# Optional variables: CC (default gcc), CUDA_HOME, RRC_NVCC (explicit nvcc path), CUDA_ARCH / ARCH,
# SAN (extra compile+link flags, used by `make asan`).

RRC_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD   ?= $(RRC_DIR)/build

ifeq ($(origin CC),default)
CC := gcc
endif

CPPFLAGS += -I$(RRC_DIR)/include -I$(RRC_DIR)/kernels_common -D_POSIX_C_SOURCE=200809L
CFLAGS   ?= -O2
CFLAGS   += -std=c11 -Wall -Wextra -Werror -fopenmp -MMD -MP
LDLIBS   += -fopenmp -lm -lpthread

LIB_SRCS  := $(wildcard $(RRC_DIR)/src/*.c)
LIB_SRCS  := $(filter-out $(RRC_DIR)/src/rrc_inspect.c,$(LIB_SRCS))
LIB_OBJS  := $(patsubst $(RRC_DIR)/src/%.c,$(BUILD)/obj/src/%.o,$(LIB_SRCS))
LIB       := $(BUILD)/librrc.a

BIN_SRCS  := $(RRC_DIR)/src/rrc_inspect.c
BINS      := $(BUILD)/rrc_inspect

# Pure-C parts of the CUDA line (GEMM plan, host emulation backend, selftest harness): gcc-built,
# linked into the C tests, rrc_selftest_host and librrc_cuda.a. No CUDA dependency.
CUDA_HOST_SRCS := $(RRC_DIR)/cuda/gemm_plan.c $(RRC_DIR)/cuda/kc_emu.c $(RRC_DIR)/cuda/selftest_ops.c
CUDA_HOST_OBJS := $(patsubst $(RRC_DIR)/cuda/%.c,$(BUILD)/obj/cuda_host/%.o,$(CUDA_HOST_SRCS))
CPPFLAGS += -I$(RRC_DIR)/cuda
CFLAGS   += $(SAN)
SELFTEST_HOST := $(BUILD)/rrc_selftest_host

TEST_SRCS := $(wildcard $(RRC_DIR)/tests/c/*.c)
TEST_OBJS := $(patsubst $(RRC_DIR)/tests/c/%.c,$(BUILD)/obj/tests/%.o,$(TEST_SRCS))
TEST_BIN  := $(BUILD)/rrc_tests

# ---- CUDA ---------------------------------------------------------------------------------
# CUDA_ARCH (or its alias ARCH, only if it looks like sm_*) selects the SASS target; PTX for the same compute capability is
# embedded too. Device math flags are spelled out: no FMA contraction (-fmad=false), IEEE division
# and square root, denormals kept, so the kernels' fp32 arithmetic is the CPU backend's (§11.3).
ifneq ($(filter sm_%,$(ARCH)),)
CUDA_ARCH := $(ARCH)
endif
CUDA_ARCH ?= sm_120
CUDA_COMPUTE := $(subst sm_,compute_,$(CUDA_ARCH))
ifneq ($(strip $(RRC_NVCC)),)
NVCC := $(RRC_NVCC)
else ifneq ($(strip $(CUDA_HOME)),)
NVCC := $(CUDA_HOME)/bin/nvcc
else
NVCC :=
endif
ifneq ($(strip $(CUDA_HOME)),)
CUDA_ROOT := $(CUDA_HOME)
else ifneq ($(strip $(RRC_NVCC)),)
CUDA_ROOT := $(abspath $(dir $(RRC_NVCC))/..)
endif
NVCC_INC := $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/include)
CUDA_LIBDIR := $(firstword $(wildcard $(CUDA_ROOT)/lib64 $(CUDA_ROOT)/lib))
NVCCFLAGS ?= -O2
NVCCFLAGS += -std=c++17 -ccbin $(CC) -arch=$(CUDA_ARCH) -gencode arch=$(CUDA_COMPUTE),code=$(CUDA_COMPUTE) \
             -fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
             -Xcompiler -Wall,-Wextra,-Werror -Werror all-warnings \
             -I$(RRC_DIR)/include -I$(RRC_DIR)/kernels_common -I$(RRC_DIR)/cuda $(NVCC_INC)
CUDA_SRCS := $(wildcard $(RRC_DIR)/cuda/*.cu)
CUDA_OBJS := $(patsubst $(RRC_DIR)/cuda/%.cu,$(BUILD)/obj/cuda/%.o,$(CUDA_SRCS))
CUDA_LIB  := $(BUILD)/librrc_cuda.a
comma := ,
CUDA_LDLIBS := $(if $(CUDA_LIBDIR),-L$(CUDA_LIBDIR) -Wl$(comma)-rpath$(comma)$(CUDA_LIBDIR)) \
               -lcublasLt -lcudart -lstdc++ -ldl -lrt

.PHONY: all cpu test cuda cuda-objs asan clean nvcc-check nvcc-missing
all: cpu

cpu: $(LIB) $(BINS)

$(BUILD)/obj/src/%.o: $(RRC_DIR)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/obj/cuda_host/%.o: $(RRC_DIR)/cuda/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/obj/tests/%.o: $(RRC_DIR)/tests/c/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/rrc_inspect: $(BUILD)/obj/src/rrc_inspect.o $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# --wrap=malloc: tests/c/test_contracts.c counts the allocation sizes of library code.
$(TEST_BIN): $(TEST_OBJS) $(CUDA_HOST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $(TEST_OBJS) $(CUDA_HOST_OBJS) $(LIB) -o $@ -Wl,--wrap=malloc $(LDLIBS)

# rrc_selftest with a host backend as the device (kc-emu by default): runs the harness without a GPU.
$(BUILD)/obj/cuda_host/selftest_host.o: $(RRC_DIR)/cuda/selftest.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DRRC_SELFTEST_HOST -c $< -o $@

$(SELFTEST_HOST): $(BUILD)/obj/cuda_host/selftest_host.o $(CUDA_HOST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# The C tests use python3 (if present) to produce a safetensors file with the reference package;
# that sub-test is skipped with a message when python3 / safetensors / torch are unavailable.
test: cpu $(TEST_BIN) $(SELFTEST_HOST)
	@rm -rf $(BUILD)/test_tmp && mkdir -p $(BUILD)/test_tmp
	RRC_TEST_TMPDIR=$(BUILD)/test_tmp $(TEST_BIN)
	$(SELFTEST_HOST) --backend emu --deferred --quick > $(BUILD)/test_tmp/selftest_host.log || { cat $(BUILD)/test_tmp/selftest_host.log; exit 1; }
	@tail -n 1 $(BUILD)/test_tmp/selftest_host.log
ifneq ($(strip $(NVCC)),)
	@$(MAKE) --no-print-directory nvcc-check
else
	@echo "nvcc header check: SKIPPED (set CUDA_HOME or RRC_NVCC to enable)"
endif

# Compiles every RRC_HD helper into a __global__ kernel for $(CUDA_ARCH) (COMPILED-ONLY check).
nvcc-check:
	@test -n "$(NVCC)" || { echo "nvcc-check: set CUDA_HOME or RRC_NVCC"; exit 1; }
	@mkdir -p $(BUILD)/obj/cuda_check
	$(NVCC) $(NVCCFLAGS) -x cu -c $(RRC_DIR)/tests/cuda/headers_check.cu -o $(BUILD)/obj/cuda_check/headers_check.o
	@echo "nvcc header check: OK ($(CUDA_ARCH) + $(CUDA_COMPUTE) PTX)"

# COMPILED-ONLY: every cuda/*.cu to an object (what CI runs; needs nvcc, no GPU, no CUDA libraries).
# The objects are plain prerequisites (no sub-make), so `make -j cuda` builds each object once.
cuda-objs: $(if $(NVCC),nvcc-check $(CUDA_OBJS),nvcc-missing)
	@echo "cuda-objs: compiled $(words $(CUDA_OBJS)) CUDA sources for $(CUDA_ARCH) + $(CUDA_COMPUTE) PTX (COMPILED-ONLY)"

nvcc-missing:
	@echo "make cuda / cuda-objs: set CUDA_HOME=/path/to/cuda (or RRC_NVCC=/path/to/nvcc)"; exit 1

# librrc_cuda.a (kernels + backend + the pure-C plan/harness objects) and rrc_selftest. Linking needs
# libcudart and libcublasLt from $(CUDA_HOME)/lib64 (or lib); running needs a GPU and its driver.
cuda: cuda-objs $(CUDA_LIB) $(BUILD)/rrc_selftest
	@echo "make cuda: built $(CUDA_LIB) and $(BUILD)/rrc_selftest (run it on the GPU: $(BUILD)/rrc_selftest)"

$(CUDA_LIB): $(CUDA_OBJS) $(CUDA_HOST_OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/obj/cuda_host/selftest.o: $(RRC_DIR)/cuda/selftest.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NVCC_INC) -c $< -o $@

$(BUILD)/rrc_selftest: $(BUILD)/obj/cuda_host/selftest.o $(CUDA_LIB) $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(CUDA_LDLIBS) $(LDLIBS)

$(BUILD)/obj/cuda/%.o: $(RRC_DIR)/cuda/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

# C tests and rrc_selftest_host under AddressSanitizer + UBSan, in a separate build directory.
asan:
	$(MAKE) --no-print-directory BUILD=$(BUILD)/asan SAN="-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g" test

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BUILD)/obj/src/rrc_inspect.d $(CUDA_HOST_OBJS:.o=.d) $(CUDA_OBJS:.o=.d)
