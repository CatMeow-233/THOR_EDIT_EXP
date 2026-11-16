# rrc — build. All outputs go to $(BUILD) (default rrc/build/, gitignored).
#
#   make cpu                 static library + binaries with the CPU backend
#   make test                build and run the C unit tests (+ nvcc header check if CUDA is set)
#   make cuda CUDA_HOME=...  build librrc_cuda.a + rrc_selftest with nvcc (CUDA_ARCH or ARCH, default sm_120)
#   make cuda-objs CUDA_HOME=...  compile every cuda/*.cu to an object only (CI, no GPU, no link)
#   make asan                `make test` (C tests, rrc_selftest_host, + cuda-host-test with CUDA_HOME)
#                            under AddressSanitizer + UBSan (build/asan)
#   make tsan                data + train C tests (prefetch thread) under ThreadSanitizer (build/tsan)
#   make cuda-host-test CUDA_HOME=...  HOST-ONLY: the real cuda/*.cu built with g++ on a fake CUDA runtime
#                            (tests/cuda_host; needs only the toolkit headers; part of `test` when CUDA_HOME is set)
#   make clean
#
# Optional variables: CC (default gcc), CUDA_HOME, RRC_NVCC (explicit nvcc path), CUDA_ARCH / ARCH,
# RRC_SAN_FLAGS (extra compile + link flags, used by `make asan` / `make tsan`; SAN is an alias).

RRC_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD   ?= $(RRC_DIR)/build

ifeq ($(origin CC),default)
CC := gcc
endif

CPPFLAGS += -I$(RRC_DIR)/include -I$(RRC_DIR)/kernels_common -D_POSIX_C_SOURCE=200809L
CFLAGS   ?= -O2
CFLAGS   += -std=c11 -Wall -Wextra -Werror -fopenmp -MMD -MP $(RRC_SAN_FLAGS)
LDLIBS   += -fopenmp -lm -lpthread $(RRC_SAN_FLAGS)

LIB_SRCS  := $(wildcard $(RRC_DIR)/src/*.c)
BIN_NAMES := rrc_inspect rrc_fwdbwd rrc_train rrc_data
LIB_SRCS  := $(filter-out $(patsubst %,$(RRC_DIR)/src/%.c,$(BIN_NAMES)),$(LIB_SRCS))
LIB_OBJS  := $(patsubst $(RRC_DIR)/src/%.c,$(BUILD)/obj/src/%.o,$(LIB_SRCS))
LIB       := $(BUILD)/librrc.a

BINS      := $(patsubst %,$(BUILD)/%,$(BIN_NAMES))

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
# toolkit headers present: the HOST-ONLY cuda-host-test can be built (used by `test`)
CUDA_HOST_OK := $(if $(CUDA_ROOT),$(wildcard $(CUDA_ROOT)/include/cublasLt.h))
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

.PHONY: all cpu test cuda cuda-objs asan tsan clean nvcc-check nvcc-missing cuda-host-test cuda-host-missing
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

$(BINS): $(BUILD)/%: $(BUILD)/obj/src/%.o $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# --wrap=malloc: tests/c/test_contracts.c counts the allocation sizes of library code.
# --wrap=rename: tests/c/test_train.c injects rename failures into checkpoint publishing.
$(TEST_BIN): $(TEST_OBJS) $(CUDA_HOST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $(TEST_OBJS) $(CUDA_HOST_OBJS) $(LIB) -o $@ -Wl,--wrap=malloc -Wl,--wrap=rename $(LDLIBS)

# rrc_selftest with a host backend as the device (kc-emu by default): runs the harness without a GPU.
$(BUILD)/obj/cuda_host/selftest_host.o: $(RRC_DIR)/cuda/selftest.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DRRC_SELFTEST_HOST -c $< -o $@

$(SELFTEST_HOST): $(BUILD)/obj/cuda_host/selftest_host.o $(CUDA_HOST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# expect_fail(command, log, what): the selftest must report FAIL itself (exit status 1 and a last log
# line 'rrc_selftest: FAIL'), not crash, abort under a sanitizer or reject its arguments.
expect_fail = st=0; $(1) > $(2) 2>&1 || st=$$?; \
	if [ $$st -ne 1 ] || [ "$$(tail -n 1 $(2))" != "rrc_selftest: FAIL" ]; then \
	  echo "expected 'rrc_selftest: FAIL' with exit 1 for: $(3) (exit $$st, log $(2))"; tail -n 5 $(2); exit 1; fi

# The C tests use python3 (if present) to produce a safetensors file with the reference package;
# that sub-test is skipped with a message when python3 / safetensors / torch are unavailable.
test: cpu $(TEST_BIN) $(SELFTEST_HOST)
	@rm -rf $(BUILD)/test_tmp && mkdir -p $(BUILD)/test_tmp
	RRC_TEST_TMPDIR=$(BUILD)/test_tmp $(TEST_BIN)
	$(SELFTEST_HOST) --backend emu --deferred --quick > $(BUILD)/test_tmp/selftest_host.log || { cat $(BUILD)/test_tmp/selftest_host.log; exit 1; }
	@tail -n 1 $(BUILD)/test_tmp/selftest_host.log
	@# negative checks of the program's exit status: no cuBLASLt (simulated: every triple refused) and
	@# an --only filter matching nothing must both fail
	@$(call expect_fail,$(SELFTEST_HOST) --quick --only gemm --reject-mask 0xFF,$(BUILD)/test_tmp/selftest_reject.log,every GEMM on the kernel fallback)
	@$(call expect_fail,$(SELFTEST_HOST) --only nonexistent_op,$(BUILD)/test_tmp/selftest_nomatch.log,no check run)
	@echo "rrc_selftest_host negative checks: OK (kernel-only GEMM and an empty --only run both fail)"
ifneq ($(strip $(NVCC)),)
	@$(MAKE) --no-print-directory nvcc-check
else
	@echo "nvcc header check: SKIPPED (set CUDA_HOME or RRC_NVCC to enable)"
endif
ifneq ($(CUDA_HOST_OK),)
	@$(MAKE) --no-print-directory cuda-host-test
else
	@echo "cuda-host-test (real cuda/*.cu on a fake runtime, HOST-ONLY): SKIPPED (set CUDA_HOME for the toolkit headers)"
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

# ---- HOST-ONLY test build of the real CUDA sources (tests/cuda_host) ---------------------------------
# cuda_kernels.cu and cuda_backend.cu compiled by g++ as C++ (no __CUDACC__) with tests/cuda_host/shim.h
# force-included, linked with a fake CUDA runtime / cuBLASLt stand-in that runs kernel launches on host
# threads. Needs the toolkit headers only (no nvcc, no CUDA libraries, no GPU). Never part of librrc_cuda.
CXX_HOST ?= g++
SHIM_DIR := $(RRC_DIR)/tests/cuda_host
SHIM_CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra -Werror -pthread $(RRC_SAN_FLAGS) $(SAN) -DRRC_CUDA_HOST_SHIM \
                 -I$(RRC_DIR)/include -I$(RRC_DIR)/kernels_common -I$(RRC_DIR)/cuda -I$(SHIM_DIR) \
                 $(if $(CUDA_ROOT),-isystem $(CUDA_ROOT)/include) -MMD -MP
SHIM_OBJS := $(BUILD)/obj/cuda_shim/cuda_kernels.o $(BUILD)/obj/cuda_shim/cuda_backend.o $(BUILD)/obj/cuda_shim/fake_cuda.o
SHIM_TEST := $(BUILD)/test_cuda_host
SHIM_SELFTEST := $(BUILD)/rrc_selftest_shim

$(BUILD)/obj/cuda_shim/%.o: $(RRC_DIR)/cuda/%.cu $(SHIM_DIR)/shim.h
	@mkdir -p $(dir $@)
	$(CXX_HOST) $(SHIM_CXXFLAGS) -include $(SHIM_DIR)/shim.h -x c++ -c $< -o $@

$(BUILD)/obj/cuda_shim/fake_cuda.o: $(SHIM_DIR)/fake_cuda.cpp $(SHIM_DIR)/shim.h $(SHIM_DIR)/fake_cuda.h
	@mkdir -p $(dir $@)
	$(CXX_HOST) $(SHIM_CXXFLAGS) -c $< -o $@

$(BUILD)/obj/cuda_shim/test_cuda_host.o: $(SHIM_DIR)/test_cuda_host.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -I$(SHIM_DIR) $(CFLAGS) -c $< -o $@

$(SHIM_TEST): $(BUILD)/obj/cuda_shim/test_cuda_host.o $(SHIM_OBJS) $(CUDA_HOST_OBJS) $(LIB)
	$(CXX_HOST) $(RRC_SAN_FLAGS) $(SAN) -pthread $^ -o $@ -fopenmp -lm

# rrc_selftest's CUDA main (selftest.c without RRC_SELFTEST_HOST) on the fake runtime
$(SHIM_SELFTEST): $(BUILD)/obj/cuda_host/selftest.o $(SHIM_OBJS) $(CUDA_HOST_OBJS) $(LIB)
	$(CXX_HOST) $(RRC_SAN_FLAGS) $(SAN) -pthread $^ -o $@ -fopenmp -lm

cuda-host-test: $(if $(CUDA_HOST_OK),$(SHIM_TEST) $(SHIM_SELFTEST),cuda-host-missing)
	@mkdir -p $(BUILD)/test_tmp
	RRC_TEST_TMPDIR=$(BUILD)/test_tmp $(SHIM_TEST)
	@# the program's CUDA main (pass logic, options) on one cheap op; test_cuda_host ran every op already
	$(SHIM_SELFTEST) --quick --only convert > $(BUILD)/test_tmp/selftest_shim.log || { cat $(BUILD)/test_tmp/selftest_shim.log; exit 1; }
	@grep -E "^(capped|grid) pass" $(BUILD)/test_tmp/selftest_shim.log; tail -n 1 $(BUILD)/test_tmp/selftest_shim.log
	@$(call expect_fail,RRC_FAKE_LT=reject $(SHIM_SELFTEST) --quick --only gemm,$(BUILD)/test_tmp/selftest_shim_reject.log,cuBLASLt refusing everything)
	@$(call expect_fail,$(SHIM_SELFTEST) --only nonexistent_op,$(BUILD)/test_tmp/selftest_shim_nomatch.log,no check run)
	@echo "cuda-host-test: OK (HOST-ONLY: real cuda/*.cu sources on a fake runtime and host threads; no GPU)"

cuda-host-missing:
	@echo "make cuda-host-test: set CUDA_HOME=/path/to/cuda (only its include/ is used)"; exit 1

# `make test` (C tests, rrc_selftest_host and its negative checks, and with CUDA_HOME the HOST-ONLY
# cuda-host-test) under AddressSanitizer + UndefinedBehaviorSanitizer, built in a separate directory
# ($(BUILD)/asan) with the normal warning/OpenMP flags plus RRC_SAN_FLAGS; any UBSan report aborts
# the run. OMP_NUM_THREADS defaults to 2 here.
ASAN_FLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
asan:
	ASAN_OPTIONS=$${ASAN_OPTIONS:-detect_leaks=1} UBSAN_OPTIONS=$${UBSAN_OPTIONS:-print_stacktrace=1} \
	  OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} \
	  $(MAKE) --no-print-directory BUILD=$(BUILD)/asan RRC_SAN_FLAGS="$(ASAN_FLAGS)" test

# The data / train suites (the prefetch thread next to the training loop) under ThreadSanitizer.
# OMP_NUM_THREADS=1: libgomp is not TSan-instrumented, so OpenMP regions would report false races.
# setarch -R disables ASLR, which some kernels need for TSan's memory layout.
TSAN_FLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=thread
tsan:
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/tsan RRC_SAN_FLAGS="$(TSAN_FLAGS)" $(BUILD)/tsan/rrc_tests
	@mkdir -p $(BUILD)/tsan/test_tmp
	for s in data train; do \
	  RRC_TEST_TMPDIR=$(BUILD)/tsan/test_tmp OMP_NUM_THREADS=1 TSAN_OPTIONS=halt_on_error=1 \
	    setarch $$(uname -m) -R $(BUILD)/tsan/rrc_tests $$s || exit 1; \
	done

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(patsubst %,$(BUILD)/obj/src/%.d,$(BIN_NAMES)) \
         $(CUDA_HOST_OBJS:.o=.d) $(CUDA_OBJS:.o=.d) $(SHIM_OBJS:.o=.d) $(BUILD)/obj/cuda_shim/test_cuda_host.d
