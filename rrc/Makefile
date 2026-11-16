# rrc — build. All outputs go to $(BUILD) (default rrc/build/, gitignored).
#
#   make cpu                 static library + binaries with the CPU backend
#   make test                build and run the C unit tests (+ nvcc header check if CUDA is set)
#   make cuda CUDA_HOME=...  compile cuda/*.cu with nvcc (CUDA_ARCH, default sm_120)
#   make clean
#
# Optional variables: CC (default gcc), CUDA_HOME, RRC_NVCC (explicit nvcc path), CUDA_ARCH.

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

.PHONY: all cpu test cuda clean nvcc-check
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

$(BUILD)/rrc_inspect: $(BUILD)/obj/src/rrc_inspect.o $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_BIN): $(TEST_OBJS) $(LIB)
	$(CC) $(CFLAGS) $(TEST_OBJS) $(LIB) -o $@ $(LDLIBS)

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

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BUILD)/obj/src/rrc_inspect.d
