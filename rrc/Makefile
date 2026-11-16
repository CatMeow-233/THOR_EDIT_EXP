# rrc — build. All outputs go to $(BUILD) (default rrc/build/, gitignored).
#
#   make cpu                 static library + binaries with the CPU backend
#   make test                build and run the C unit tests (+ nvcc header check if CUDA is set)
#   make cuda CUDA_HOME=...  compile cuda/*.cu with nvcc (CUDA_ARCH, default sm_120)
#   make asan                C tests under AddressSanitizer + UBSan (build/asan)
#   make clean
#
# Optional variables: CC (default gcc), CUDA_HOME, RRC_NVCC (explicit nvcc path), CUDA_ARCH,
# RRC_SAN_FLAGS (extra compile + link flags, used by `make asan`).

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

TEST_SRCS := $(wildcard $(RRC_DIR)/tests/c/*.c)
TEST_OBJS := $(patsubst $(RRC_DIR)/tests/c/%.c,$(BUILD)/obj/tests/%.o,$(TEST_SRCS))
TEST_BIN  := $(BUILD)/rrc_tests

# ---- CUDA ---------------------------------------------------------------------------------
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
NVCC_INC := -I$(CUDA_HOME)/include
else ifneq ($(strip $(RRC_NVCC)),)
NVCC_INC := -I$(abspath $(dir $(RRC_NVCC))/../include)
endif
NVCCFLAGS ?= -O2
NVCCFLAGS += -std=c++17 -ccbin $(CC) -arch=$(CUDA_ARCH) -gencode arch=$(CUDA_COMPUTE),code=$(CUDA_COMPUTE) \
             -Xcompiler -Wall,-Wextra,-Werror -Werror all-warnings \
             -I$(RRC_DIR)/include -I$(RRC_DIR)/kernels_common $(NVCC_INC)
CUDA_SRCS := $(wildcard $(RRC_DIR)/cuda/*.cu)
CUDA_OBJS := $(patsubst $(RRC_DIR)/cuda/%.cu,$(BUILD)/obj/cuda/%.o,$(CUDA_SRCS))

.PHONY: all cpu test cuda clean nvcc-check asan
all: cpu

cpu: $(LIB) $(BINS)

$(BUILD)/obj/src/%.o: $(RRC_DIR)/src/%.c
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
$(TEST_BIN): $(TEST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $(TEST_OBJS) $(LIB) -o $@ -Wl,--wrap=malloc $(LDLIBS)

# The C tests use python3 (if present) to produce a safetensors file with the reference package;
# that sub-test is skipped with a message when python3 / safetensors / torch are unavailable.
test: cpu $(TEST_BIN)
	@rm -rf $(BUILD)/test_tmp && mkdir -p $(BUILD)/test_tmp
	RRC_TEST_TMPDIR=$(BUILD)/test_tmp $(TEST_BIN)
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

# Placeholder for the CUDA backend (later stage): compiles cuda/*.cu when present.
cuda:
	@test -n "$(NVCC)" || { echo "make cuda: set CUDA_HOME=/path/to/cuda (or RRC_NVCC=/path/to/nvcc)"; exit 1; }
	@$(MAKE) --no-print-directory nvcc-check
ifeq ($(strip $(CUDA_SRCS)),)
	@echo "make cuda: no cuda/*.cu sources yet (CUDA backend is a later stage)"
else
	@$(MAKE) --no-print-directory $(CUDA_OBJS)
endif

$(BUILD)/obj/cuda/%.o: $(RRC_DIR)/cuda/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# C tests under AddressSanitizer + UndefinedBehaviorSanitizer, built in a separate directory
# ($(BUILD)/asan) with the normal warning/OpenMP flags plus RRC_SAN_FLAGS; any UBSan report aborts
# the run. OMP_NUM_THREADS defaults to 2 here.
ASAN_FLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
asan:
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/asan RRC_SAN_FLAGS="$(ASAN_FLAGS)" cpu $(BUILD)/asan/rrc_tests
	@mkdir -p $(BUILD)/asan/test_tmp
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} \
	  RRC_TEST_TMPDIR=$(BUILD)/asan/test_tmp $(BUILD)/asan/rrc_tests

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(patsubst %,$(BUILD)/obj/src/%.d,$(BIN_NAMES))
