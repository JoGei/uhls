CALL_DIR ?= $(if $(PWD),$(abspath $(PWD)),$(CURDIR))
BUILD_DIR ?= $(CALL_DIR)/build

export UHLS_ROOT ?= $(CURDIR)/..
UHLS ?= $(UHLS_ROOT)/uhls
DOT ?= dot
YOSYS ?= yosys

STEM ?= dot4_relu
SRC ?= $(STEM)/$(STEM).c
TOP ?= $(STEM)
RES ?= $(UHLS_ROOT)/src/uhls/backend/hls/impl/generic/gen.uhlslib.json
VENDOR_RES ?= $(UHLS_ROOT)/src/uhls/backend/hls/impl/vendor/ihp130/ihp130.uhlslib.json

LEGALIZE_PASSES ?= inline,prune,const_prop
LEGALIZE_PASSES_ARGS ?= --pass-arg $(TOP)
OPT_PASSES ?= dce,cse,copy_prop,dce
OPT_PASSES_ARGS ?=
OPT2_PASSES ?=
OPT2_PASSES_ARGS ?=
GOPT_PASSES ?= infer_loops,translate_loop_dialect,infer_static,simplify_static_control,predicate,fold_predicates
GOPT_PASSES_ARGS ?=
ALLOC_MEM ?= autoram+1024
SCHED_ALGO ?= asap
BIND_ALGO ?= left_edge
FSM_ENCODING ?= binary
RTL_HDL ?= verilog
RTL_EXT ?= v
DRV_LANG ?= c
DRV_EXT ?= h
RUN_BACKEND ?= verilator
ASIC_TARGET ?= ihp130
ASIC_OUTDIR ?= $(BUILD_DIR)/asic

# slave | master | none
GLUE_WRAP ?= slave
# wishbone | obi
GLUE_PROTOCOL ?= obi

GLUE_WRAP_NORM := $(strip $(GLUE_WRAP))
GLUE_PROTOCOL_NORM := $(strip $(GLUE_PROTOCOL))
ALLOC_MEM_NORM := $(strip $(ALLOC_MEM))
VENDOR_RES_NORM := $(strip $(VENDOR_RES))

ALLOC_ARGS := -exg $(RES)
ifneq ($(ALLOC_MEM_NORM),)
ALLOC_ARGS += --mem=$(ALLOC_MEM_NORM)
endif
ifneq ($(VENDOR_RES_NORM),)
ALLOC_ARGS += --vendor=$(VENDOR_RES_NORM)
endif

GLUE_ARGS :=
ifneq ($(GLUE_WRAP_NORM),)
GLUE_ARGS += --wrap=$(GLUE_WRAP_NORM)
endif
ifneq ($(GLUE_PROTOCOL_NORM),)
GLUE_ARGS += --protocol=$(GLUE_PROTOCOL_NORM)
endif

UIR := $(BUILD_DIR)/$(STEM).uir
LEGALIZE_UIR := $(BUILD_DIR)/$(STEM).legalize.uir
OPT_UIR := $(BUILD_DIR)/$(STEM).opt.uir
OPT2_UIR := $(BUILD_DIR)/$(STEM).opt2.uir
SEQ_UHIR := $(BUILD_DIR)/$(STEM).uhir
GOPT_UHIR := $(BUILD_DIR)/$(STEM).opt.uhir
ALLOC_UHIR := $(BUILD_DIR)/$(STEM).alloc.uhir
SCHED_UHIR := $(BUILD_DIR)/$(STEM).sched.uhir
BIND_UHIR := $(BUILD_DIR)/$(STEM).bind.uhir
FSM_UHIR := $(BUILD_DIR)/$(STEM).fsm.uhir
UGLIR := $(BUILD_DIR)/$(STEM).uglir
RTL := $(BUILD_DIR)/$(STEM).$(RTL_EXT)
PROT_WRAP_UGLIR := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).uglir
PROT_WRAP_RTL := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).$(RTL_EXT)
PROT_WRAP_DRV := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).$(DRV_EXT)
PROT_WRAP_MMIO := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).mmio
PROT_WRAP_MMIO_DOT := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).mmio.dot
PROT_WRAP_MMIO_SVG := $(BUILD_DIR)/$(STEM).$(GLUE_PROTOCOL_NORM)_$(GLUE_WRAP_NORM).mmio.svg
ASIC_TOP := $(ASIC_OUTDIR)/$(STEM).v
ASIC_SDC := $(ASIC_OUTDIR)/constraints.sdc
ASIC_ORFS := $(ASIC_OUTDIR)/orfs.mk
ASIC_MACROS := $(ASIC_OUTDIR)/macros.json
ASIC_FLOORPLAN_HINTS := $(ASIC_OUTDIR)/floorplan_hints.tcl
ASIC_MACRO_PLACE := $(ASIC_OUTDIR)/macro_place.tcl
ASIC_ORFS_DIR := $(ASIC_OUTDIR)/orfs
ASIC_ORFS_CONFIG := $(ASIC_ORFS_DIR)/config.mk
ASIC_ORFS_RUN := $(ASIC_ORFS_DIR)/run_orfs.sh
ASIC_YOSYS_LOG := $(ASIC_OUTDIR)/yosys_smoke.log
ASIC_YOSYS_STAMP := $(ASIC_OUTDIR)/.yosys_smoke.stamp
ASIC_ORFS_STAMP := $(ASIC_OUTDIR)/.orfs_run.stamp
ASIC_STAMP := $(ASIC_OUTDIR)/.stamp
ASIC_ORFS_BASE := $(ASIC_ORFS_DIR)/results/ihp-sg13g2/$(STEM)/base
ASIC_ROUTE_ODB := $(ASIC_ORFS_BASE)/5_1_grt.odb
ASIC_CTS_ODB := $(ASIC_ORFS_BASE)/4_1_cts.odb
ASIC_PLACE_ODB := $(ASIC_ORFS_BASE)/3_5_place_dp.odb
ASIC_PDN_ODB := $(ASIC_ORFS_BASE)/2_4_floorplan_pdn.odb
ASIC_FLOORPLAN_ODB := $(ASIC_ORFS_BASE)/2_1_floorplan.odb
ASIC_YOSYS_NETLIST := $(ASIC_ORFS_BASE)/1_2_yosys.v
ASIC_SCHEM_PREFIX := $(ASIC_OUTDIR)/$(STEM)_synth
ASIC_SCHEM_DOT := $(ASIC_SCHEM_PREFIX).dot
ASIC_SCHEM_SVG := $(ASIC_SCHEM_PREFIX).svg

CFG_DOT := $(BUILD_DIR)/$(STEM).cfg.dot
CFG_SVG := $(BUILD_DIR)/$(STEM).cfg.svg
DFG_DOT := $(BUILD_DIR)/$(STEM).dfg.dot
DFG_SVG := $(BUILD_DIR)/$(STEM).dfg.svg
CDFG_DOT := $(BUILD_DIR)/$(STEM).cdfg.dot
CDFG_SVG := $(BUILD_DIR)/$(STEM).cdfg.svg
SEQ_DOT := $(BUILD_DIR)/$(STEM).seq.dot
SEQ_SVG := $(BUILD_DIR)/$(STEM).seq.svg
GOPT_DOT := $(BUILD_DIR)/$(STEM).gopt.dot
GOPT_SVG := $(BUILD_DIR)/$(STEM).gopt.svg
ALLOC_DOT := $(BUILD_DIR)/$(STEM).alloc.dot
ALLOC_SVG := $(BUILD_DIR)/$(STEM).alloc.svg
BIND_DOT := $(BUILD_DIR)/$(STEM).bind.dot
BIND_SVG := $(BUILD_DIR)/$(STEM).bind.svg
BIND_COMPAT_DOT := $(BUILD_DIR)/$(STEM).bind.compat.dot
BIND_COMPAT_SVG := $(BUILD_DIR)/$(STEM).bind.compat.svg
BIND_CONFLICT_DOT := $(BUILD_DIR)/$(STEM).bind.conflict.dot
BIND_CONFLICT_SVG := $(BUILD_DIR)/$(STEM).bind.conflict.svg
BIND_TRP_DOT := $(BUILD_DIR)/$(STEM).bind.trp.dot
BIND_TRP_SVG := $(BUILD_DIR)/$(STEM).bind.trp.svg
FSM_DOT := $(BUILD_DIR)/$(STEM).fsm.dot
FSM_SVG := $(BUILD_DIR)/$(STEM).fsm.svg

PIPELINE_ARTIFACTS := $(UIR) $(LEGALIZE_UIR) $(OPT_UIR) $(OPT2_UIR) $(SEQ_UHIR) $(GOPT_UHIR) $(ALLOC_UHIR) $(SCHED_UHIR) $(BIND_UHIR) $(FSM_UHIR) $(UGLIR) $(RTL)
WRAPPED_PIPELINE_ARTIFACTS := $(PROT_WRAP_UGLIR) $(PROT_WRAP_RTL) $(PROT_WRAP_DRV)
VIEW_ARTIFACTS := $(CFG_SVG) $(DFG_SVG) $(CDFG_SVG) $(SEQ_SVG) $(GOPT_SVG) $(ALLOC_SVG) $(BIND_SVG) $(BIND_COMPAT_SVG) $(BIND_CONFLICT_SVG) $(BIND_TRP_SVG) $(FSM_SVG) $(PROT_WRAP_MMIO) $(PROT_WRAP_MMIO_SVG)
ASIC_ARTIFACTS := $(ASIC_TOP) $(ASIC_SDC) $(ASIC_ORFS) $(ASIC_MACROS) $(ASIC_FLOORPLAN_HINTS) $(ASIC_MACRO_PLACE) $(ASIC_ORFS_CONFIG) $(ASIC_ORFS_RUN)

.PHONY: all pipeline synth frontend middleend backend wrapped_backend wrapped views asic yosys_smoke orfs_run asic_gui asic_schem run_verilator clean

all: synth run_hw_interpret

yosys: yosys_smoke

openroad: orfs_run

pipeline: synth

synth: $(PIPELINE_ARTIFACTS) $(WRAPPED_PIPELINE_ARTIFACTS)

frontend: $(UIR)

middleend: $(OPT2_UIR)

backend: $(RTL)

wrapped_backend: $(PROT_WRAP_RTL)

wrapped: $(WRAPPED_PIPELINE_ARTIFACTS)

views: $(VIEW_ARTIFACTS)

asic: $(ASIC_ARTIFACTS) $(ASIC_ORFS_STAMP)

yosys_smoke: $(ASIC_YOSYS_STAMP)

orfs_run: $(ASIC_ORFS_STAMP)

asic_gui: $(ASIC_ORFS_STAMP)
	OPENROAD_BIN="$${OPENROAD_EXE:-}" ; \
	if [ -z "$$OPENROAD_BIN" ] && [ -n "$$ORFS_ROOT" ] && [ -x "$$ORFS_ROOT/tools/install/OpenROAD/bin/openroad" ]; then \
		OPENROAD_BIN="$$ORFS_ROOT/tools/install/OpenROAD/bin/openroad" ; \
	fi ; \
	if [ -z "$$OPENROAD_BIN" ]; then \
		OPENROAD_BIN="$$(command -v openroad || true)" ; \
	fi ; \
	if [ -z "$$OPENROAD_BIN" ]; then \
		echo "error: could not resolve openroad; set OPENROAD_EXE or ORFS_ROOT, or put openroad on PATH" >&2 ; \
		exit 1 ; \
	fi ; \
	ODB="" ; \
	for candidate in $(ASIC_ROUTE_ODB) $(ASIC_CTS_ODB) $(ASIC_PLACE_ODB) $(ASIC_PDN_ODB) $(ASIC_FLOORPLAN_ODB); do \
		if [ -f "$$candidate" ]; then \
			ODB="$$candidate" ; \
			break ; \
		fi ; \
	done ; \
	if [ -z "$$ODB" ]; then \
		echo "error: no ORFS .odb checkpoint found under $(ASIC_ORFS_BASE)" >&2 ; \
		exit 1 ; \
	fi ; \
	echo "Opening $$ODB" ; \
	"$$OPENROAD_BIN" -gui -db "$$ODB"

asic_schem: $(ASIC_SCHEM_SVG)

run_hw_interpret: $(UIR) $(PROT_WRAP_UGLIR)
	$(UHLS) run $(UIR) --backend=$(RUN_BACKEND) --uglir $(PROT_WRAP_UGLIR)

$(BUILD_DIR):
	mkdir -p $@

$(UIR): $(SRC) | $(BUILD_DIR)
	$(UHLS) parse $< -o $@

$(LEGALIZE_UIR): $(UIR) | $(BUILD_DIR)
	if [ -n "$(strip $(LEGALIZE_PASSES))" ]; then \
		$(UHLS) opt $< -p $(LEGALIZE_PASSES) $(LEGALIZE_PASSES_ARGS) -o $@; \
	else \
		cp $< $@; \
	fi

$(OPT_UIR): $(LEGALIZE_UIR) | $(BUILD_DIR)
	if [ -n "$(strip $(OPT_PASSES))" ]; then \
		$(UHLS) opt $< -p $(OPT_PASSES) $(OPT_PASSES_ARGS) -o $@; \
	else \
		cp $< $@; \
	fi

$(OPT2_UIR): $(OPT_UIR) | $(BUILD_DIR)
	if [ -n "$(strip $(OPT2_PASSES))" ]; then \
		$(UHLS) opt $< -p $(OPT2_PASSES) $(OPT2_PASSES_ARGS) -o $@; \
	else \
		cp $< $@; \
	fi

$(CFG_DOT): $(OPT2_UIR) | $(BUILD_DIR)
	$(UHLS) view $< --what cfg --dot -o $@

$(CFG_SVG): $(CFG_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(DFG_DOT): $(OPT2_UIR) | $(BUILD_DIR)
	$(UHLS) view $< --what dfg --dot -o $@

$(DFG_SVG): $(DFG_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(CDFG_DOT): $(OPT2_UIR) | $(BUILD_DIR)
	$(UHLS) view $< --what cdfg --dot -o $@

$(CDFG_SVG): $(CDFG_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(SEQ_UHIR): $(OPT2_UIR) | $(BUILD_DIR)
	$(UHLS) seq $< --top $(TOP) -o $@

$(SEQ_DOT): $(SEQ_UHIR) | $(BUILD_DIR)
	$(UHLS) view $< --dot -o $@

$(SEQ_SVG): $(SEQ_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(GOPT_UHIR): $(SEQ_UHIR) | $(BUILD_DIR)
	$(UHLS) gopt $< -p $(GOPT_PASSES) $(GOPT_PASSES_ARGS) -o $@

$(GOPT_DOT): $(GOPT_UHIR) | $(BUILD_DIR)
	$(UHLS) view $(GOPT_UHIR) --dot -o $@

$(GOPT_SVG): $(GOPT_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(ALLOC_UHIR): $(GOPT_UHIR) $(RES) $(VENDOR_RES) | $(BUILD_DIR)
	$(UHLS) alloc $< $(ALLOC_ARGS) -o $@

$(ALLOC_DOT): $(ALLOC_UHIR) | $(BUILD_DIR)
	$(UHLS) view $(ALLOC_UHIR) --dot -o $@

$(ALLOC_SVG): $(ALLOC_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(SCHED_UHIR): $(ALLOC_UHIR) | $(BUILD_DIR)
	$(UHLS) sched --algo $(SCHED_ALGO) $< -o $@

$(BIND_UHIR): $(SCHED_UHIR) | $(BUILD_DIR)
	$(UHLS) bind $< --algo $(BIND_ALGO) --flatten -o $@

$(BIND_DOT): $(BIND_UHIR) | $(BUILD_DIR)
	$(UHLS) view $< --what dfgsb_unroll --dot -o $@

$(BIND_SVG): $(BIND_DOT) | $(BUILD_DIR)
	$(DOT) -Lg -Tsvg $< -o $@

$(BIND_COMPAT_DOT): $(BIND_UHIR) | $(BUILD_DIR)
	$(UHLS) view $< --what compatibility --dot -o $@

$(BIND_COMPAT_SVG): $(BIND_COMPAT_DOT) | $(BUILD_DIR)
	$(DOT) -Lg -Tsvg $< -o $@

$(BIND_CONFLICT_DOT): $(BIND_UHIR) | $(BUILD_DIR)
	$(UHLS) view $< --what conflict --dot -o $@

$(BIND_CONFLICT_SVG): $(BIND_CONFLICT_DOT) | $(BUILD_DIR)
	$(DOT) -Lg -Tsvg $< -o $@

$(BIND_TRP_DOT): $(BIND_UHIR) | $(BUILD_DIR)
	$(UHLS) view $< --what trp_unroll --dot -o $@

$(BIND_TRP_SVG): $(BIND_TRP_DOT) | $(BUILD_DIR)
	$(DOT) -Lg -Tsvg $< -o $@

$(FSM_UHIR): $(BIND_UHIR) | $(BUILD_DIR)
	$(UHLS) fsm $< --encoding=$(FSM_ENCODING) -o $@

$(FSM_DOT): $(FSM_UHIR) | $(BUILD_DIR)
	$(UHLS) view $(FSM_UHIR) --dot -o $@

$(FSM_SVG): $(FSM_DOT) | $(BUILD_DIR)
	$(DOT) -Lg -Tsvg $< -o $@

$(UGLIR): $(FSM_UHIR) | $(BUILD_DIR)
	$(UHLS) glue $(GLUE_ARGS) $< -o $@

$(RTL): $(UGLIR) | $(BUILD_DIR)
	$(UHLS) rtl $< --hdl=$(RTL_HDL) -o $@

$(PROT_WRAP_UGLIR): $(FSM_UHIR) | $(BUILD_DIR)
	$(UHLS) glue --wrap=$(GLUE_WRAP_NORM) --protocol=$(GLUE_PROTOCOL_NORM) $< -o $@

$(PROT_WRAP_RTL): $(PROT_WRAP_UGLIR) | $(BUILD_DIR)
	$(UHLS) rtl $< --hdl=$(RTL_HDL) -o $@

$(PROT_WRAP_DRV): $(PROT_WRAP_UGLIR) | $(BUILD_DIR)
	$(UHLS) drv $< --lang=$(DRV_LANG) -o $@

$(PROT_WRAP_MMIO): $(PROT_WRAP_UGLIR) | $(BUILD_DIR)
	$(UHLS) view $< --what mmio -o $@

$(PROT_WRAP_MMIO_DOT): $(PROT_WRAP_UGLIR) | $(BUILD_DIR)
	$(UHLS) view $< --what mmio --dot -o $@

$(PROT_WRAP_MMIO_SVG): $(PROT_WRAP_MMIO_DOT) | $(BUILD_DIR)
	$(DOT) -Tsvg $< -o $@

$(ASIC_STAMP): $(PROT_WRAP_UGLIR) | $(BUILD_DIR)
	mkdir -p $(ASIC_OUTDIR)
	$(UHLS) asic $< --target=$(ASIC_TARGET) --outdir $(ASIC_OUTDIR)
	touch $@

$(ASIC_TOP): $(ASIC_STAMP)

$(ASIC_SDC): $(ASIC_STAMP)

$(ASIC_ORFS): $(ASIC_STAMP)

$(ASIC_MACROS): $(ASIC_STAMP)

$(ASIC_FLOORPLAN_HINTS): $(ASIC_STAMP)

$(ASIC_MACRO_PLACE): $(ASIC_STAMP)

$(ASIC_ORFS_CONFIG): $(ASIC_STAMP)

$(ASIC_ORFS_RUN): $(ASIC_STAMP)

$(ASIC_YOSYS_STAMP): $(ASIC_STAMP)
	cd $(ASIC_ORFS_DIR) && \
	set -euo pipefail && \
	VERILOG_FILES="$$(sed -n 's/^export VERILOG_FILES = //p' config.mk)" && \
	VERILOG_INCLUDE_DIRS="$$(sed -n 's/^export VERILOG_INCLUDE_DIRS = //p' config.mk)" && \
	VERILOG_DEFINES="$$(sed -n 's/^export VERILOG_DEFINES = //p' config.mk)" && \
	ADDITIONAL_LIBS="$$(sed -n 's/^export ADDITIONAL_LIBS = //p' config.mk)" && \
	TOP_MODULE="$$(sed -n 's/^export TOP_MODULE = //p' config.mk)" && \
	INC_FLAGS="" && \
	LIBERTY_CMD="" && \
	for dir in $$VERILOG_INCLUDE_DIRS; do \
		INC_FLAGS="$$INC_FLAGS -I$$dir"; \
	done && \
	if [ -n "$$ADDITIONAL_LIBS" ]; then \
		LIBERTY_CMD="read_liberty -lib $$ADDITIONAL_LIBS; "; \
	fi && \
	$(YOSYS) -p "$$LIBERTY_CMD read_verilog $$INC_FLAGS $$VERILOG_DEFINES $$VERILOG_FILES; hierarchy -check -top $$TOP_MODULE; proc; memory; opt; check; stat" | tee ../$(notdir $(ASIC_YOSYS_LOG))
	touch $@

$(ASIC_ORFS_STAMP): $(ASIC_STAMP)
	cd $(ASIC_ORFS_DIR) && ./run_orfs.sh
	touch $@

$(ASIC_SCHEM_SVG): $(ASIC_ORFS_STAMP)
	$(YOSYS) -p "read_verilog $(ASIC_YOSYS_NETLIST); hierarchy -top $(TOP); show -viewer none -format svg -prefix $(ASIC_SCHEM_PREFIX)"

clean:
	rm -rf $(BUILD_DIR)
