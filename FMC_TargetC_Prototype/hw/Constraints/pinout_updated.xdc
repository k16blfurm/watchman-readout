# ----------------------------------------------------------------------------
# PINOUT Constraint File
# ----------------------------------------------------------------------------
# For SOMB board

set_property PULLUP true [get_ports SDA]
set_property PULLUP true [get_ports SCL]
# ----------------------------------------------------------------------------
# FMC Expansion Connector - Bank 34
# ----------------------------------------------------------------------------

set_property PACKAGE_PIN Y14 [get_ports BB3]
set_property PACKAGE_PIN U14 [get_ports BB4]

set_property PACKAGE_PIN P19 [get_ports {HCMPB[0]}]
set_property PACKAGE_PIN N18 [get_ports {HCMPA[0]}]
set_property PACKAGE_PIN P20 [get_ports {HCMPC[0]}]
set_property PACKAGE_PIN N20 [get_ports {HCMPD[0]}]
#set_property PULLUP true [get_ports {HCMPB}];
#set_property PULLUP true [get_ports {HCMPA}];
#set_property PULLUP true [get_ports {HCMPC}];
#set_property PULLUP true [get_ports {HCMPD}];


set_property PACKAGE_PIN B20 [get_ports TRIGA] # changed from W14 to B20
set_property PACKAGE_PIN C20 [get_ports TRIGB] # changed from U15 to C20
set_property PACKAGE_PIN G14 [get_ports TRIGC] # changed from R16 to G14 
set_property PACKAGE_PIN J15 [get_ports TRIGD] # changed from M18 to J15



set_property PACKAGE_PIN T11 [get_ports MONTIMING_P] # change from Y16 to T11
set_property PACKAGE_PIN T10 [get_ports MONTIMING_N] # changed from Y17 to T10


set_property PACKAGE_PIN D18 [get_ports SS_LD_DIR] # changed from U17 to D18
set_property PACKAGE_PIN E18 [get_ports SS_LD_SIN] # changed from T16 to E18
set_property PACKAGE_PIN W15 [get_ports REGCLR]


set_property PACKAGE_PIN Y13 [get_ports DO_5] # change from W20 to Y13
set_property PACKAGE_PIN V5 [get_ports DO_8] # changed from Y19 to V5
set_property PACKAGE_PIN W6 [get_ports DO_7] # changed from Y18 to W6
set_property PACKAGE_PIN L20 [get_ports DO_1] # change from W16 to L20
set_property PACKAGE_PIN E17 [get_ports RAMP] # changed from V16 to E17
set_property PACKAGE_PIN E19 [get_ports SS_RESET] # changed from R17 to E19
set_property PACKAGE_PIN J16 [get_ports DO_10] # changed from R18 to J16 
set_property PACKAGE_PIN V6 [get_ports DO_6] # changed from T17 to V6
set_property PACKAGE_PIN L14 [get_ports DO_13] # changed from V18 to L14
set_property PACKAGE_PIN L15 [get_ports DO_12] # changed from V17 to L15
#set_property PACKAGE_PIN W19 [get_ports DOE]
set_property PACKAGE_PIN M14 [get_ports DO_2] # change from W18 to M14
set_property PACKAGE_PIN M15 [get_ports DO_3] # changed from P18 to M15
set_property PACKAGE_PIN L19 [get_ports DONE] # changed from N17 to L19


# ----------------------------------------------------------------------------
# FMC Expansion Connector - Bank 35
# ----------------------------------------------------------------------------


set_property PACKAGE_PIN H17 [get_ports BB2]
set_property PACKAGE_PIN K16 [get_ports DO_11] # changed from H16 to K16
set_property PACKAGE_PIN H18 [get_ports RDAD_CLK] #changed H18 from H18
set_property PACKAGE_PIN F19 [get_ports RDAD_SIN] #changed from J18 to F19
set_property PACKAGE_PIN G15 [get_ports DO_14] # changed from D18 to G15
set_property PACKAGE_PIN E17 [get_ports BB1]
set_property PACKAGE_PIN V10 [get_ports SS_INCR] # changed from D20 to V10
set_property PACKAGE_PIN Y12 [get_ports DO_4] # changed from D19 to Y12
set_property PACKAGE_PIN E19 [get_ports BB5]
set_property PACKAGE_PIN H15 [get_ports DO_15] # changed from E18 to H15
set_property PACKAGE_PIN H20 [get_ports DO_16] # changed from F17 to H20
set_property PACKAGE_PIN V11 [get_ports DO_9] # changed from F16 to V11
set_property PACKAGE_PIN K17 [get_ports WR_CS_S4] # changed from L20 to K17
set_property PACKAGE_PIN F20 [get_ports RDAD_DIR] #changed from L19 to F20
set_property PACKAGE_PIN J20 [get_ports SAMPLESEL_ANY] # changed from M17 to J20
set_property PACKAGE_PIN K19 [get_ports WR_CS_S2] # changed from J19 to K19
set_property PACKAGE_PIN J19 [get_ports WR_CS_S3] # changed from K19 to J19
set_property PACKAGE_PIN M19 [get_ports WR_CS_S0] # changed from  G18 to M19
set_property PACKAGE_PIN M20 [get_ports WR_CS_S1] # changed from G17 to M20
set_property PACKAGE_PIN J18 [get_ports GCC_RESET] # changed from F20 to J18
set_property PACKAGE_PIN K18 [get_ports WR_CS_S5] # changed from F19 to K18
set_property PACKAGE_PIN F16 [get_ports WR_RS_S0] # changed from H20 to F16
set_property PACKAGE_PIN F17 [get_ports WR_RS_S1] # changed from J20 to F17
set_property PACKAGE_PIN D9 [get_ports SDA] #changed from G15 to D9
set_property PACKAGE_PIN C6 [get_ports SCL] # changed from H15 to C6
set_property PACKAGE_PIN D19 [get_ports PCLK] #changed from N16 to D19
set_property PACKAGE_PIN D20 [get_ports SHOUT] #changed from N15 to D20
set_property PACKAGE_PIN T19 [get_ports SIN] # changed from L15 to T19
set_property PACKAGE_PIN P14 [get_ports SCLK] #changed from L14 to P14


# ----------------------------------------------------------------------------
# IOSTANDARD Constraints
#
# Note that these IOSTANDARD constraints are applied to all IOs currently
# assigned within an I/O bank.  If these IOSTANDARD constraints are
# evaluated prior to other PACKAGE_PIN constraints being applied, then
# the IOSTANDARD specified will likely not be applied properly to those
# pins.  Therefore, bank wide IOSTANDARD constraints should be placed
# within the XDC file in a location that is evaluated AFTER all
# PACKAGE_PIN constraints within the target bank have been evaluated.
#
# Un-comment one or more of the following IOSTANDARD constraints according to
# the bank pin assignments that are required within a design.
# ----------------------------------------------------------------------------

# Set the bank voltage for IO Bank 34 to 3.3V by default.
# set_property IOSTANDARD LVCMOS33 [get_ports -of_objects [get_iobanks 34]];

# Set the bank voltage for IO Bank 35 to 3.3V by default.
# set_property IOSTANDARD LVCMOS33 [get_ports -of_objects [get_iobanks 35]];

# Set the bank voltage for IO Bank 13 to 3.3V by default. (I/O bank available on Z7020 device only)
# set_property IOSTANDARD LVCMOS33 [get_ports -of_objects [get_iobanks 13]];

#IOSTANDARD
#BANK 34
set_property IOSTANDARD LVCMOS25 [get_ports {HCMPA[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {HCMPB[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {HCMPC[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {HCMPD[0]}]

set_property IOSTANDARD LVCMOS25 [get_ports TRIGA]
set_property IOSTANDARD LVCMOS25 [get_ports BB3]
set_property IOSTANDARD LVCMOS25 [get_ports SS_LD_SIN]
set_property IOSTANDARD LVCMOS25 [get_ports SS_LD_DIR]
set_property IOSTANDARD LVCMOS25 [get_ports REGCLR]
set_property IOSTANDARD LVCMOS25 [get_ports BB4]
set_property IOSTANDARD LVCMOS25 [get_ports TRIGB]
set_property IOSTANDARD LVCMOS25 [get_ports DO_5]
set_property IOSTANDARD LVCMOS25 [get_ports DO_7]
set_property IOSTANDARD LVCMOS25 [get_ports DO_8]
set_property IOSTANDARD LVCMOS25 [get_ports RAMP]
set_property IOSTANDARD LVCMOS25 [get_ports DO_1]
set_property IOSTANDARD LVCMOS25 [get_ports TRIGC] 
set_property IOSTANDARD LVCMOS25 [get_ports SS_RESET]
set_property IOSTANDARD LVCMOS25 [get_ports DO_6]
set_property IOSTANDARD LVCMOS25 [get_ports DO_10]
set_property IOSTANDARD LVCMOS25 [get_ports DO_12]
set_property IOSTANDARD LVCMOS25 [get_ports DO_13]
set_property IOSTANDARD LVCMOS25 [get_ports DO_2]
#set_property IOSTANDARD LVCMOS25 [get_ports DOE]
set_property IOSTANDARD LVCMOS25 [get_ports DONE]
set_property IOSTANDARD LVCMOS25 [get_ports DO_3]

set_property IOSTANDARD LVDS_25 [get_ports SSTIN_P]
set_property IOSTANDARD LVDS_25 [get_ports SSTIN_N]
set_property PACKAGE_PIN T12 [get_ports SSTIN_P] # changed from P14 to T12
set_property PACKAGE_PIN U12 [get_ports SSTIN_N] # changed from R14 to U12
set_property IOSTANDARD LVDS_25 [get_ports MONTIMING_P]
set_property IOSTANDARD LVDS_25 [get_ports MONTIMING_N]


#BANK 35
set_property IOSTANDARD LVCMOS25 [get_ports DO_11]
set_property IOSTANDARD LVCMOS25 [get_ports BB2]
set_property IOSTANDARD LVCMOS25 [get_ports RDAD_SIN] 
set_property IOSTANDARD LVCMOS25 [get_ports RDAD_CLK] 
set_property IOSTANDARD LVCMOS25 [get_ports BB1]
set_property IOSTANDARD LVCMOS25 [get_ports DO_14]
set_property IOSTANDARD LVCMOS25 [get_ports DO_4]
set_property IOSTANDARD LVCMOS25 [get_ports SS_INCR]
set_property IOSTANDARD LVCMOS25 [get_ports DO_15]
set_property IOSTANDARD LVCMOS25 [get_ports BB5]
set_property IOSTANDARD LVCMOS25 [get_ports DO_9]
set_property IOSTANDARD LVCMOS25 [get_ports DO_16]
set_property IOSTANDARD LVCMOS25 [get_ports RDAD_DIR]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S4]
set_property IOSTANDARD LVCMOS25 [get_ports SAMPLESEL_ANY]
set_property IOSTANDARD LVCMOS25 [get_ports TRIGD]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S2]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S3]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S0]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S1]
set_property IOSTANDARD LVCMOS25 [get_ports WR_CS_S5]
set_property IOSTANDARD LVCMOS25 [get_ports GCC_RESET]
set_property IOSTANDARD LVCMOS25 [get_ports WR_RS_S1] 
set_property IOSTANDARD LVCMOS25 [get_ports WR_RS_S0] 
set_property IOSTANDARD LVCMOS25 [get_ports SCL]
set_property IOSTANDARD LVCMOS25 [get_ports SDA]
set_property IOSTANDARD LVCMOS25 [get_ports SHOUT]
set_property IOSTANDARD LVCMOS25 [get_ports PCLK]
set_property IOSTANDARD LVCMOS25 [get_ports SCLK]
set_property IOSTANDARD LVCMOS25 [get_ports SIN]


set_property IOSTANDARD LVDS_25 [get_ports HSCLK_P]
set_property IOSTANDARD LVDS_25 [get_ports HSCLK_N] 
set_property PACKAGE_PIN B19 [get_ports HSCLK_P] #changed from K14 to B19
set_property PACKAGE_PIN A20 [get_ports HSCLK_N] #changed from J14 to to A20
set_property IOSTANDARD LVDS_25 [get_ports WL_CLK_P]
set_property IOSTANDARD LVDS_25 [get_ports WL_CLK_N]
set_property PACKAGE_PIN V12 [get_ports WL_CLK_P] # changed from M19 to V12
set_property PACKAGE_PIN W13 [get_ports WL_CLK_N] # changed from M20 to W13




