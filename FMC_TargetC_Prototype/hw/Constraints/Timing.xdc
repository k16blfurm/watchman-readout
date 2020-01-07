
# ----------------------------------------------------------------------------
# TIMING Constraint File
# ----------------------------------------------------------------------------


set_property async_reg true [get_cells -hierarchical *cdc_reg*]
set_false_path -from [get_clocks *] -to [get_cells -hierarchical *cdc_reg*]

set_property async_reg true [get_cells -hierarchical *DFF_AB*]
#set_false_path -from [get_clocks *] -to [get_cells -hierarchical *DFF_AB*]

set_property async_reg true [get_cells -hierarchical *DFF_B*]
set_false_path -from [get_clocks *] -to [get_pins -hierarchical -filter Name=~*DFF_AB/Q_reg/D]

#set_property DRIVE 16 [get_ports BB3]
#set_property SLEW FAST [get_ports BB3]
set_property DIFF_TERM TRUE [get_ports MONTIMING_N]
set_property DIFF_TERM TRUE [get_ports MONTIMING_P]














create_debug_core u_ila_0 ila
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
set_property ALL_PROBE_SAME_MU_CNT 1 [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets [list base_zynq_i/processing_system7_0/inst/FCLK_CLK0]]
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe0]
set_property port_width 9 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets [list {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[0]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[1]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[2]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[3]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[4]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[5]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[6]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[7]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/window2read[8]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe1]
set_property port_width 9 [get_debug_ports u_ila_0/probe1]
connect_debug_port u_ila_0/probe1 [get_nets [list {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[0]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[1]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[2]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[3]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[4]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[5]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[6]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[7]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/wr_shifted[8]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe2]
set_property port_width 9 [get_debug_ports u_ila_0/probe2]
connect_debug_port u_ila_0/probe2 [get_nets [list {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[0]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[1]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[2]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[3]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[4]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[5]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[6]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[7]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/WDOSTORE/WdoNumber[8]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe3]
set_property port_width 2 [get_debug_ports u_ila_0/probe3]
connect_debug_port u_ila_0/probe3 [get_nets [list {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_RS[0]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_RS[1]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe4]
set_property port_width 6 [get_debug_ports u_ila_0/probe4]
connect_debug_port u_ila_0/probe4 [get_nets [list {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[0]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[1]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[2]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[3]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[4]} {base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/WR_CS[5]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe5]
set_property port_width 1 [get_debug_ports u_ila_0/probe5]
connect_debug_port u_ila_0/probe5 [get_nets [list base_zynq_i/TARGET_C_TopLevel_Sy_0/U0/TC_RoundBuffer/circBuffer/trigger]]
set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets u_ila_0_FCLK_CLK0]
