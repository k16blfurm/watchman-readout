# ----------------------------------------------------------------------------
#     _____
#    /     \
#   /____   \____
#  / \===\   \==/
# /___\===\___\/  AVNET Design Resource Center
#      \======/         www.em.avnet.com/drc
#       \====/    
# ----------------------------------------------------------------------------
# 
#  Created With Avnet UCF Generator V0.4.0 
#     Date: Wednesday, November 27, 2013 
#     Time: 2:10:18 PM 
# 
#  This design is the property of Avnet.  Publication of this
#  design is not authorized without written consent from Avnet.
#  
#  Please direct any questions or issues to the MicroZed Community Forums:
#     http://www.microzed.org
# 
#  Disclaimer:
#     Avnet, Inc. makes no warranty for the use of this code or design.
#     This code is provided  "As Is". Avnet, Inc assumes no responsibility for
#     any errors, which may appear in this code, nor does it make a commitment
#     to update the information contained herein. Avnet, Inc specifically
#     disclaims any implied warranties of fitness for a particular purpose.
#                      Copyright(c) 2013 Avnet, Inc.
#                              All rights reserved.
# 
# ----------------------------------------------------------------------------
#
#  Notes:
# 
#  27 November 2013
#     IO standards based upon Bank 34, Bank 35 Vcco supply options of 1.8V, 
#     2.5V, or 3.3V are possible based upon the Vadj jumper (J18) settings.  
#     By default, Vadj is expected to be set to 1.8V but if a different 
#     voltage is used for a particular design, then the corresponding IO 
#     standard within this UCF should also be updated to reflect the actual 
#     Vadj jumper selection.
# 
#     Net names are not allowed to contain hyphen characters '-' since this
#     is not a legal VHDL87 or Verilog character within an identifier.  
#     HDL net names are adjusted to contain no hyphen characters '-' but 
#     rather use underscore '_' characters.  Comment net name with the hyphen 
#     characters will remain in place since these are intended to match the 
#     schematic net names in order to better enable schematic search.
#
#     In the following, the XDC constraint is matched to the origanal UCF 
#     constraint, UCF commented out above, XDC equivalent placedbelow the UCF.
#
# ----------------------------------------------------------------------------
 
# Bank 13, Vcco = Vadj 
# Set the bank voltage for bank 13.
set_property IOSTANDARD LVCMOS18 [get_ports -filter { IOBANK == 13 } ]

set_property PACKAGE_PIN V7  [get_ports {BANK13_LVDS_0_N}] # JX1-89 A.RDAD_dir
set_property PACKAGE_PIN U7  [get_ports {BANK13_LVDS_0_P}] # JX1-87 A.SAMPLESEL_AN
set_property PACKAGE_PIN U10 [get_ports {BANK13_LVDS_1_N}] # JX1-90 A.WR_CS_S4
set_property PACKAGE_PIN T9  [get_ports {BANK13_LVDS_1_P}] # JX1-88 A.WR_CS_S3
set_property PACKAGE_PIN W8  [get_ports {BANK13_LVDS_2_N}] # JX1-93 A.RDAD_clk
set_property PACKAGE_PIN V8  [get_ports {BANK13_LVDS_2_P}] # JX1-91 A.RDAD_sin
set_property PACKAGE_PIN U5  [get_ports {BANK13_LVDS_3_N}] # JX1-94 A.GCC_Reset
set_property PACKAGE_PIN T5  [get_ports {BANK13_LVDS_3_P}] # JX1-92 A.WR_CS_S5
set_property PACKAGE_PIN Y13 [get_ports {BANK13_LVDS_4_N}] # JX2-95 B.DO_5
set_property PACKAGE_PIN Y12 [get_ports {BANK13_LVDS_4_P}] # JX2-93 B.DO_4
set_property PACKAGE_PIN V10 [get_ports {BANK13_LVDS_5_N}] # JX2-96 B.SS_Incr
set_property PACKAGE_PIN V11 [get_ports {BANK13_LVDS_5_P}] # JX2-94 B.D0_9
set_property PACKAGE_PIN W6  [get_ports {BANK13_LVDS_6_N}] # JX2-99 B.DO_7
set_property PACKAGE_PIN V6  [get_ports {BANK13_LVDS_6_P}] # JX2-97 B.DO_6
set_property PACKAGE_PIN V5  [get_ports {BANK13_SE_0}] # B.DO_8

# Bank 34, Vcco = Vadj
# Set the bank voltage for bank 34.
set_property IOSTANDARD LVCMOS18 [get_ports -filter { IOBANK == 34 } ]

set_property PACKAGE_PIN T10 [get_ports {JX1_LVDS_0_N}]  # jx1-13 Montiming_N                   
set_property PACKAGE_PIN T11 [get_ports {JX1_LVDS_0_P}]  # JX1-11 MONTIMING_P
set_property PACKAGE_PIN U12 [get_ports {JX1_LVDS_1_N}]  # JX1-14 SSTIN_N
set_property PACKAGE_PIN T12 [get_ports {JX1_LVDS_1_P}]  # JX1-12 SSTI_P
set_property PACKAGE_PIN U15 [get_ports {JX1_LVDS_10_N}] # JX1-43 A.DO_5       srcc
set_property PACKAGE_PIN U14 [get_ports {JX1_LVDS_10_P}] # JX1-41 A.DO_4       srcc
set_property PACKAGE_PIN U19 [get_ports {JX1_LVDS_11_N}] # JX1-44 Bit_clk_P    mrcc  ADC       inv
set_property PACKAGE_PIN U18 [get_ports {JX1_LVDS_11_P}] # JX1-42 Bit_clk_N    mrcc  ADC      inv
set_property PACKAGE_PIN P19 [get_ports {JX1_LVDS_12_N}] # JX1-49 A.DO.7
set_property PACKAGE_PIN N18 [get_ports {JX1_LVDS_12_P}] # JX1-47 A.DO_6
set_property PACKAGE_PIN P20 [get_ports {JX1_LVDS_13_N}] # JX1-50 A.Trig2
set_property PACKAGE_PIN N20 [get_ports {JX1_LVDS_13_P}] # JX1-48 A.Trig3
set_property PACKAGE_PIN U20 [get_ports {JX1_LVDS_14_N}] # JX1-55 A.SS_Incr
set_property PACKAGE_PIN T20 [get_ports {JX1_LVDS_14_P}] # JX1-53 A.DO_8
set_property PACKAGE_PIN W20 [get_ports {JX1_LVDS_15_N}] # JX1-56 A.WR_RS_S0 
set_property PACKAGE_PIN V20 [get_ports {JX1_LVDS_15_P}] # JX1-54 RESET    ADC
set_property PACKAGE_PIN Y19 [get_ports {JX1_LVDS_16_N}] # JX1-63 A.DO_10
set_property PACKAGE_PIN Y18 [get_ports {JX1_LVDS_16_P}] # JX1-61 A.DO_9
set_property PACKAGE_PIN W16 [get_ports {JX1_LVDS_17_N}] # JX1-64 A.WR_CS_S0
set_property PACKAGE_PIN V16 [get_ports {JX1_LVDS_17_P}] # JX1-54 A.WR_RS_S1
set_property PACKAGE_PIN R17 [get_ports {JX1_LVDS_18_N}] # JX1-69 A.DO_12
set_property PACKAGE_PIN R16 [get_ports {JX1_LVDS_18_P}] # JX1-67 A.DO_11
set_property PACKAGE_PIN R18 [get_ports {JX1_LVDS_19_N}] # JX1-70 CH1.outB_P         ADC      inv
set_property PACKAGE_PIN T17 [get_ports {JX1_LVDS_19_P}] # JX1-68 CH1.outB_N         ADC      inv
set_property PACKAGE_PIN V13 [get_ports {JX1_LVDS_2_N}]  # JX1-19 A.SS_LD_dir     
set_property PACKAGE_PIN U13 [get_ports {JX1_LVDS_2_P}]  # JX1-17 A.RAMP
set_property PACKAGE_PIN V18 [get_ports {JX1_LVDS_20_N}] # JX1-75 A.DO_14
set_property PACKAGE_PIN V17 [get_ports {JX1_LVDS_20_P}] # JX1-73 A.DO_13
set_property PACKAGE_PIN W19 [get_ports {JX1_LVDS_21_N}] # JX1-76  CH1.OUTA_P        ADC      inv
set_property PACKAGE_PIN W18 [get_ports {JX1_LVDS_21_P}] # JX1-74 CH1.OUTA_N         ADC      inv
set_property PACKAGE_PIN P18 [get_ports {JX1_LVDS_22_N}] # JX1-83 A.DO_16
set_property PACKAGE_PIN N17 [get_ports {JX1_LVDS_22_P}] # JX1-81 A.DO_15
set_property PACKAGE_PIN P16 [get_ports {JX1_LVDS_23_N}] # JX1-84 A.WR_CS_S2
set_property PACKAGE_PIN P15 [get_ports {JX1_LVDS_23_P}] # JX1-82 A.WR_CS_S1
set_property PACKAGE_PIN W13 [get_ports {JX1_LVDS_3_N}]  # JX1-20 WL_CLK_N
set_property PACKAGE_PIN V12 [get_ports {JX1_LVDS_3_P}]  # JX1-18 WL_CLK_P
set_property PACKAGE_PIN T15 [get_ports {JX1_LVDS_4_N}]  # JX1-25 A.SS_Reset
set_property PACKAGE_PIN T14 [get_ports {JX1_LVDS_4_P}]  # JX1-23 A.SS_LD_sin
set_property PACKAGE_PIN R14 [get_ports {JX1_LVDS_5_N}]  # JX1-26 A.PCLK 
set_property PACKAGE_PIN P14 [get_ports {JX1_LVDS_5_P}]  # JX1-24 SCLK 
set_property PACKAGE_PIN Y17 [get_ports {JX1_LVDS_6_N}]  # JX1-31 A.DO_1
set_property PACKAGE_PIN Y16 [get_ports {JX1_LVDS_6_P}]  # JX1-29 A.DONE
set_property PACKAGE_PIN Y14 [get_ports {JX1_LVDS_7_N}]  # JX1-32 A.Trig4
set_property PACKAGE_PIN W14 [get_ports {JX1_LVDS_7_P}]  # JX1-30 A.SHOUT
set_property PACKAGE_PIN U17 [get_ports {JX1_LVDS_8_N}]  # JX1-37 A.DO_3
set_property PACKAGE_PIN T16 [get_ports {JX1_LVDS_8_P}]  # JX1-35 A.DO_2
set_property PACKAGE_PIN W15 [get_ports {JX1_LVDS_9_N}]  # JX1-38 A.HSCLK_N
set_property PACKAGE_PIN V15 [get_ports {JX1_LVDS_9_P}]  # JX1-36 A.HSCLK_P
set_property PACKAGE_PIN R19 [get_ports {JX1_SE_0}]      # JX1-9 A.Trig1
set_property PACKAGE_PIN T19 [get_ports {JX1_SE_1}]      # JX1-10 SIN

# Bank 35, Vcco = Vadj
# Set the bank voltage for bank 35.
set_property IOSTANDARD LVCMOS18 [get_ports -filter { IOBANK == 35 } ]

set_property PACKAGE_PIN B20 [get_ports {JX2_LVDS_0_N}]   # JX2-19 B.Trig1
set_property PACKAGE_PIN C20 [get_ports {JX2_LVDS_0_P}]   # JX2-17 B.Trig2
set_property PACKAGE_PIN A20 [get_ports {JX2_LVDS_1_N}]   # JX2-18 B.Hsclk_N
set_property PACKAGE_PIN B19 [get_ports {JX2_LVDS_1_P}]   # JX2-20 B.Hsclk_P
set_property PACKAGE_PIN L17 [get_ports {JX2_LVDS_10_N}]  # JX2-49 SyncClk_P    inv
set_property PACKAGE_PIN L16 [get_ports {JX2_LVDS_10_P}]  # JX2-47 SyncClk_N    inv
set_property PACKAGE_PIN K18 [get_ports {JX2_LVDS_11_N}]  # JX2-50 B.WR_CS_S5
set_property PACKAGE_PIN K17 [get_ports {JX2_LVDS_11_P}]  # JX2-48 B.WR_CS_S4
set_property PACKAGE_PIN H17 [get_ports {JX2_LVDS_12_N}]  # JX2-55 SyncTrig_P   inv    
set_property PACKAGE_PIN H16 [get_ports {JX2_LVDS_12_P}]  # JX2-53 SyncTrig_N  
set_property PACKAGE_PIN H18 [get_ports {JX2_LVDS_13_N}]  # JX2-56 B.RDAD_clk
set_property PACKAGE_PIN J18 [get_ports {JX2_LVDS_13_P}]  # JX2-54 B.GCC_Reset
set_property PACKAGE_PIN G18 [get_ports {JX2_LVDS_14_N}]  # JX2-63 Frame_clk_N
set_property PACKAGE_PIN G17 [get_ports {JX2_LVDS_14_P}]  # JX2-61 Frame_clk_P
set_property PACKAGE_PIN F20 [get_ports {JX2_LVDS_15_N}]  # JX2-64 B.RDAD_dir
set_property PACKAGE_PIN F19 [get_ports {JX2_LVDS_15_P}]  # JX2-62 B.RDAD_sin
set_property PACKAGE_PIN G20 [get_ports {JX2_LVDS_16_N}]  # JX2-69 CH2.outA_N
set_property PACKAGE_PIN G19 [get_ports {JX2_LVDS_16_P}]  # JX2-67 CH2.outB_P
set_property PACKAGE_PIN H20 [get_ports {JX2_LVDS_17_N}]  # JX2-70 B.DO_16
set_property PACKAGE_PIN J20 [get_ports {JX2_LVDS_17_P}]  # JX2-68 B.SAMPLESEL_ANY
set_property PACKAGE_PIN J14 [get_ports {JX2_LVDS_18_N}]  # JX2-75 CH2.outB_N
set_property PACKAGE_PIN K14 [get_ports {JX2_LVDS_18_P}]  # JX2-73 CH2.outB_P
set_property PACKAGE_PIN G15 [get_ports {JX2_LVDS_19_N}]  # JX2-76 B.DO_14
set_property PACKAGE_PIN H15 [get_ports {JX2_LVDS_19_P}]   # JX2-74 B.DO_15
set_property PACKAGE_PIN D18 [get_ports {JX2_LVDS_2_N}]    #JX2-25 B.SS_LD_Dir
set_property PACKAGE_PIN E17 [get_ports {JX2_LVDS_2_P}]    #JX2-23 B.RAMP
set_property PACKAGE_PIN N16 [get_ports {JX2_LVDS_20_N}]   #JX83 Input_clk_P    inv
set_property PACKAGE_PIN N15 [get_ports {JX2_LVDS_20_P}]   #JX81 Input_clk_N     inv
set_property PACKAGE_PIN L15 [get_ports {JX2_LVDS_21_N}]   # JX-82 B.DO_13
set_property PACKAGE_PIN L14 [get_ports {JX2_LVDS_21_P}]   # JX-84 B.DO_12
set_property PACKAGE_PIN M15 [get_ports {JX2_LVDS_22_N}]   # JX2-89 B.DO_3
set_property PACKAGE_PIN M14 [get_ports {JX2_LVDS_22_P}]   # JX2-87 B.DO_2
set_property PACKAGE_PIN J16 [get_ports {JX2_LVDS_23_N}]   # JX2-90 B.DO_10
set_property PACKAGE_PIN K16 [get_ports {JX2_LVDS_23_P}]   # JX2-88 B.DO_11
set_property PACKAGE_PIN D20 [get_ports {JX2_LVDS_3_N}]    #JX2-26 B.SHOUT
set_property PACKAGE_PIN D19 [get_ports {JX2_LVDS_3_P}]    # JX2-24 B.PCLK
set_property PACKAGE_PIN E19 [get_ports {JX2_LVDS_4_N}]    # JX2-31 B.SS_Reset
set_property PACKAGE_PIN E18 [get_ports {JX2_LVDS_4_P}]    # JX2-29  B.SS_LD_dir
set_property PACKAGE_PIN F17 [get_ports {JX2_LVDS_5_N}]    # JX2-32 B.WR_RS_S1
set_property PACKAGE_PIN F16 [get_ports {JX2_LVDS_5_P}]    # JX2-30 B.WR_RS_S0
set_property PACKAGE_PIN L20 [get_ports {JX2_LVDS_6_N}]    # JX2-37 B.DO_1
set_property PACKAGE_PIN L19 [get_ports {JX2_LVDS_6_P}]    # JX2-35 B.DONE
set_property PACKAGE_PIN M20 [get_ports {JX2_LVDS_7_N}]    # JX2-38 B.WR_CS_S1
set_property PACKAGE_PIN M19 [get_ports {JX2_LVDS_7_P}]    # JX2-36 B.WR_CS_S0
set_property PACKAGE_PIN M18 [get_ports {JX2_LVDS_8_N}]    # JX2-43 SyncAck_P
set_property PACKAGE_PIN M17 [get_ports {JX2_LVDS_8_P}]    # JX2-41 SyncAck_N
set_property PACKAGE_PIN J19 [get_ports {JX2_LVDS_9_N}]    # JX2-42 B.WR_CS_S2
set_property PACKAGE_PIN K19 [get_ports {JX2_LVDS_9_P}]    # JX2-44 B.WR_CS_S3
set_property PACKAGE_PIN G14 [get_ports {JX2_SE_0}]        # JX2-13 B.Trig3
set_property PACKAGE_PIN J15 [get_ports {JX2_SE_1}]        # JX2-SE_1 B.Trig4
