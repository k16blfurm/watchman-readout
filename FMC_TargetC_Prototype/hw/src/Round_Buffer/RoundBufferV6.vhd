-- -----------------------------------------------------------
--!@FILE: 	RoundBufferV5.vhd
--!@AUTHOR: Jonathan Hendriks
--!@DATE: 	21st of January 2019
-- -----------------------------------------------------------
--!@DESCRIPTION:
--!
-- -----------------------------------------------------------
--!@VERSION:
--!
-- -----------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.TARGETC_pkg.all;
use work.WindowCPU_pkg.all;

entity RoundBufferV6 is
--	generic(
--		NBRWINDOWS : integer := 16
--	);
	port(
	--	nrst : 			in	std_Logic;
		ClockBus:		in T_ClockBus;
		Timecounter:	in std_logic_vector(63 downto 0);
		Timestamp:		in T_timestamp;
		--GrayTimeCnt:	in std_logic_vector(63 downto 0);

		trigger : 		in std_logic_vector(3 downto 0);

		WR_RS_S:		out std_logic_vector(1 downto 0);
		WR_CS_S:		out std_logic_vector(5 downto 0);

		CtrlBus_IxSL:		in 	T_CtrlBus_IxSL; --Outputs from Control Master

		--Output For control
		RBNbrOfPackets: out std_logic_vector(7 downto 0);
		-- RBNbrOfPackets:
		--CtrlBus_OxSL:		out	T_CtrlBus_OxSL; --Outputs from Control Master

		-- FIFO out for Reading RDAD
		RDAD_ReadEn  :in  std_logic;
		RDAD_DataOut : out std_logic_vector(8 downto 0);
		RDAD_Empty	: out std_logic;

		-- FIFO for FiFoManager
		AXI_ReadEn:	in	std_logic;
		AXI_Time_DataOut : out std_logic_vector(63 downto 0);
		AXI_WdoAddr_DataOut : out std_logic_vector(8 downto 0);
		AXI_TrigInfo_DataOut : out std_logic_vector(11 downto 0);
		AXI_Spare_DataOut :	out std_logic_vector(10 downto 0);
		AXI_Empty	: out std_logic;

		-- FIFO IN for Digiting
	    DIG_Full	: out	std_logic;
	    DIG_DataIn	: in	std_logic_vector(8 downto 0);
	    DIG_WriteEn	: in	std_logic;
	    
	    -- Signal for trigger the acquisition for debugging
	    address_is_zero_out : out std_logic

	    
	);
end RoundBufferV6;

architecture implementation of RoundBufferV6 is



component DummyTrigger is

    port (

  clk :            in  std_logic;
  RST :             in  std_logic;
  start_reg:	    in 	std_logic; 
  Timestamp:        in T_timestamp;

  trigger :         out std_logic

);

end component DummyTrigger;



component circularBuffer is

    port (
    
  clk :            in  std_logic;
  RST :             in  std_logic;  
  trigger :         in std_logic;
  full_fifo :        in std_logic;          
   windowStorage:     in std_logic; 
 -- ptr_window :      out std_logic_vector(8 downto 0);
 -- sstin :           out std_logic;
--   wr:     out unsigned(8 downto 0);
  enable_write :    out std_logic;
    enable_write_fifo :    out std_logic;

--  counter :          out std_logic_vector(2 downto 0);        
  
  RD_add:           out std_logic_vector(8 downto 0);
    RD_add_fifo:           out std_logic_vector(8 downto 0);

            
  WR_RS:            out std_logic_vector(1 downto 0);

  WR_CS:            out std_logic_vector(5 downto 0);
    Timestamp:        in T_timestamp

  

);
end component circularBuffer;



	component WindowBrainV2 is
		generic(
			ADDRESS : integer := 0
		);
		Port (

		nrst : 			in	std_Logic;
		nCLR :			in	std_Logic;
	--	SSTIN:			in std_logic;
		CLK:			in 	std_logic;

		--Window Part
		CPUBus :		in	std_logic_vector(10 downto 0);

		-- wr1_en:			out	std_logic;
		-- wr2_en:			out	std_logic;

		--CurAddr:		in 	std_logic_vector(7 downto 0);
		--RealAddrBit:		in 	std_logic;
		--OldAddr:		in	std_logic_vector(7 downto 0);
		--OldAddrBit:		in std_logic;

		PREVBus_IN :	in	std_logic;
		PREVBus_OUT :	out	std_logic;
		NEXTBus_IN :	in	std_logic;
		NEXTBus_OUT :	out	std_logic;

		NextAddr : 		out std_logic;
		PrevAddr :		out std_logic
		-- NextAddr : 		inout 	std_logic_vector(7 downto 0);
		-- PrevAddr :		inout 	std_logic_vector(7 downto 0)
		);

	end component WindowBrainV2;


	component WindowStoreV4 is
--		Generic(
--			NBRWINDOWS : integer := 16
--		);
		Port (
		nrst : 			in	std_Logic;
		ClockBus:		in T_ClockBus;

		ValidData:		in	std_logic;

		-- Overwatch of Transmission
		NbrOfPackets:	out	std_logic_vector(7 downto 0);
		Reg_Clr:		in	std_logic;

		CPUBus:			in 	std_logic_vector(10 downto 0);
		CPUTime:		in	T_timestamp;
		TriggerInfo:	in 	std_logic_vector(11 downto 0);
		trigger:        in  std_logic_vector(3 downto 0);
		-- Control Signals
        CtrlBus_IxSL:    in     T_CtrlBus_IxSL;

		-- FIFO out for Reading RDAD
	    RDAD_ReadEn  :in  std_logic;
	    RDAD_DataOut : out std_logic_vector(8 downto 0);
	    RDAD_Empty	: out std_logic;
        RDAD_Data_trig : in std_logic_vector(8 downto 0);
        RDAD_WriteEn_trig: in std_logic;
		-- FIFO for FiFoManager
		AXI_ReadEn:	in	std_logic;
		AXI_Time_DataOut : out std_logic_vector(63 downto 0);
		AXI_WdoAddr_DataOut : out std_logic_vector(8 downto 0);
		AXI_TrigInfo_DataOut : out std_logic_vector(11 downto 0);
		AXI_Spare_DataOut :	out std_logic_vector(10 downto 0);
	    AXI_Empty	: out std_logic
		);
	end component;


	--component CPU_CONTROLLER is
	component CPU_CONTROLLERV3 is
--				Generic(
--			NBRWINDOWS : integer := 16
--		);
		Port (
		nrst : 			in	STD_Logic;

		ClockBus:		in T_ClockBus;
		--Timecounter:	in std_logic_vector(63 downto 0);
		Timestamp:		in T_timestamp;
		trigger : 		in std_logic_vector(3 downto 0);

		-- Interface to WindowCPU
		CPUBus:			out	std_logic_vector(10 downto 0);
		CPUTime:		out	T_Timestamp;
		TriggerInfo :	out std_logic_vector(11 downto 0);

		WR_RS_S:		out std_logic_vector(1 downto 0);
		WR_CS_S:		out std_logic_vector(5 downto 0);

		CtrlBus_IxSL:		in 	T_CtrlBus_IxSL;

		ValidReal:		out std_logic;
		ValidData:		out std_logic;
		--CurAddr:		out	std_logic_vector(7 downto 0);
		--RealAddrBit:	out std_logic_vector(NBRWINDOWS-1 downto 0);
		--OldAddr:		out	std_logic_vector(7 downto 0);
		--OldAddrBit:		out std_logic_vector(NBRWINDOWS-1 downto 0);

		NextAddr_in : 	in std_logic_vector(7 downto 0);
		PrevAddr_in :	in std_logic_vector(7 downto 0);
		NextValid_in : 	in std_logic;
		PrevValid_in :	in std_logic;

	    -- FIFO IN for Digiting
	    DIG_Full	: out	std_logic;
	    DIG_DataIn	: in	std_logic_vector(8 downto 0);
	    DIG_WriteEn	: in	std_logic
		);

	--end component CPU_CONTROLLER;
end component CPU_CONTROLLERV3;

	component BIT_SELECTOR is
			Port (
			nrst :		in 	std_logic;
			nclr :		in	std_logic;
			clk : 		in 	std_logic;
			data : 		in 	std_logic_vector(255 downto 0);
			valid : 	out std_logic;
			addr :		out	std_logic_vector(7 downto 0)
			);
	end component BIT_SELECTOR;
    
    component nextAddressCnt is
     
     Port ( 
               clk : in STD_LOGIC;
               rst : in STD_LOGIC;
               usrRst : in STD_LOGIC;
               Ctrl : in STD_LOGIC;
               nextAddress : out STD_LOGIC_vector(7 downto 0);
               nextAddressValid : out STD_LOGIC;
               address_is_zero : out std_logic

               );
 
     end component nextAddressCnt;
     
      component SyncBit is 
           generic (
              SYNC_STAGES_G  : integer := 2;
              CLK_POL_G      : std_logic := '1';
              RST_POL_G      : std_logic := '1';
              INIT_STATE_G   : std_logic := '0';
              GATE_DELAY_G   : time := 1 ns
           );
           port ( 
              -- Clock and reset
              clk         : in  std_logic;
              rst         : in  std_logic := '0';
              -- Incoming bit, asynchronous
              asyncBit    : in  std_logic;
              -- Outgoing bit, synced to clk
              syncBit     : out std_logic
           ); 
        end component;
        
     
	-- -------------------------------------------------------------
	-- SIGNALS
	-- -------------------------------------------------------------

	signal TrigInfo_intl:		std_logic_vector(11 downto 0);
	signal Bus_intl :			std_logic_vector(10 downto 0);


--	signal PREVBus_intl :		std_logic_vector(NBRWINDOWS-1 downto 0);
--	signal NEXTBus_intl :		std_logic_vector(NBRWINDOWS-1 downto 0);
--	signal PREVBus_intl_Delay :		std_logic_vector(NBRWINDOWS-1 downto 0);
--	signal NEXTBus_intl_Delay :		std_logic_vector(NBRWINDOWS-1 downto 0);

--	signal NextAddrBus : 		std_logic_vector(NBRWINDOWS-1 downto 0);
--	signal PrevAddrBus :		std_logic_vector(NBRWINDOWS-1 downto 0);

	signal NextValid_s, PrevValid_s : std_logic;
	signal NextAddr_s,PrevAddr_s : std_logic_vector(7 downto 0);

	-- signal NextAddrBus : 		std_logic_vector(7 downto 0);
	-- signal PrevAddrBus :		std_logic_vector(7 downto 0);

	signal CurWindowCnt :	std_logic_vector(7 downto 0);
	signal OldWindowCnt :	std_logic_vector(7 downto 0);
	signal NextWindowCnt:	std_logic_vector(7 downto 0);
	signal PrevWindowCnt:	std_logic_vector(7 downto 0);

	--signal CtrlBus_IxSL_intl : T_CtrlBus_OxMS_Intl;
	--
	-- signal wr1_en_bus :				std_logic_vector(NBRWINDOWS-1 downto 0);
	-- signal wr2_en_bus :				std_logic_vector(NBRWINDOWS-1 downto 0);
	--
	-- signal wr1_en_single :				std_logic;
	-- signal wr2_en_single :				std_logic;

	signal Trig_intl, Last_intl, Te_intl, LE_intl, TrigDly_intl : std_logic;

	--signal RealAddrBit_s:		std_logic_vector(NBRWINDOWS-1 downto 0);  COMMENTED ON MAY 2019
	--signal OldAddrBit_s:		std_logic_vector(NBRWINDOWS-1 downto 0);  COMMENTED ON MAY 2019

	signal clr_intl : std_logic;
	signal ValidData_s, ValidReal_s : std_logic;
	signal time_intl:			T_timestamp;
    signal nrst : std_logic;
    
    signal dummytrigger_s : std_logic;

    signal RDAD_Data_s : std_logic_vector(8 downto 0);
    signal RDAD_Data_fifo_s : std_logic_vector(8 downto 0);
    
    signal RDAD_Full_s:  std_logic;

    signal RDAD_WrEn_s : std_logic;
    signal RDAD_WrEn_fifo_s : std_logic;
    signal WR_RS_S_trig:		std_logic_vector(1 downto 0);
    signal WR_CS_S_trig:        std_logic_vector(5 downto 0);
    signal WR_RS_S_user:		std_logic_vector(1 downto 0);
    signal WR_CS_S_user:        std_logic_vector(5 downto 0);
    signal trigger_s:        std_logic;

--    signal windowStorage_s : std_logic;
    
	-- -------------------------------------------------------------
	-- Constraints on Signals
	-- -------------------------------------------------------------
	attribute DONT_TOUCH : string;
--	attribute DONT_TOUCH of GEN_CPU: label is "TRUE";
--	attribute DONT_TOUCH of CPU0: label is "TRUE";
--	attribute DONT_TOUCH of CPULast: label is "TRUE";
--	attribute DONT_TOUCH of CPU_CONTROLLER_inst: label is "TRUE";
--	attribute DONT_TOUCH of WDOTRIGGER: label is "TRUE";
--	attribute DONT_TOUCH of WDOSTORE: label is "TRUE";

--	attribute DONT_TOUCH of PREVBus_intl: signal is "TRUE";
--	attribute DONT_TOUCH of NEXTBus_intl: signal is "TRUE";

	attribute DONT_TOUCH of CurWindowCnt: signal is "TRUE";
	attribute DONT_TOUCH of OldWindowCnt: signal is "TRUE";

--	attribute DONT_TOUCH of NextAddrBus: signal is "TRUE";
--	attribute DONT_TOUCH of PrevAddrBus: signal is "TRUE";

	-- attribute DONT_TOUCH of wr1_en_bus: signal is "TRUE";
	-- attribute DONT_TOUCH of wr2_en_bus: signal is "TRUE";
	-- attribute DONT_TOUCH of wr1_en_single: signal is "TRUE";
	-- attribute DONT_TOUCH of wr2_en_single: signal is "TRUE";

	attribute DONT_TOUCH of NextAddr_s : signal is "TRUE";
	attribute DONT_TOUCH of PrevAddr_s : signal is "TRUE";
	attribute DONT_TOUCH of NextValid_s : signal is "TRUE";
	attribute DONT_TOUCH of PrevValid_s : signal is "TRUE";
begin



inst_DummyTriger: DummyTrigger

    port map (

  clk =>            ClockBus.Clk125MHz ,
  RST =>            nrst,
  start_reg=>     CtrlBus_IxSL.WindowStorage,
  Timestamp=>        Timestamp,
 
  trigger =>       dummytrigger_s   

);



circBuffer: circularBuffer 

    port map(
    
  clk    =>          ClockBus.CLK125MHz  ,
  RST    =>             nrst,  
  trigger   =>        trigger_s ,
  full_fifo   =>      RDAD_Full_s  ,    
  windowStorage=>     CtrlBus_IxSL.WindowStorage,       
  enable_write  =>   RDAD_WrEn_s ,  -- For fifo to pass RD_ADD
    enable_write_fifo  =>   RDAD_WrEn_fifo_s ,  -- For fifo to pass RD_ADD

  RD_add    =>          RDAD_Data_s ,
    RD_add_fifo    =>          RDAD_Data_fifo_s ,

  WR_RS    =>           WR_RS_S_trig,
  WR_CS   =>            WR_CS_S_trig,
  Timestamp=>         Timestamp
 -- ptr_window   =>         ,
  -- sstin =>           out std_logic,
 --   wr=>     out unsigned(8 downto 0);
 --  counter   =>          out std_logic_vector(2 downto 0),
);


SyncBitNrst: SyncBit
       generic map (
          SYNC_STAGES_G  => 2,
          CLK_POL_G      => '1',
          RST_POL_G      => '1',
          INIT_STATE_G   => '0',
          GATE_DELAY_G   => 1 ns
       )
       
       port map ( 
          -- Clock and reset
          clk  => ClockBus.CLK125MHz,
          rst   => '0',
          -- Incoming bit, asynchronous
          asyncBit =>  CtrlBus_IxSL.SW_nRST,
          -- Outgoing bit, synced to clk
          syncBit   => nrst
       ); 
--nrst <= CtrlBus_IxSL.SW_nRST;



	-- wr1_en_single <= wr1_en_bus(to_integer(unsigned(OldWindowCnt)));
	-- wr2_en_single <= wr2_en_bus(to_integer(unsigned(OldWindowCnt)));

	--CPU_CONTROLLER_inst : CPU_CONTROLLER
	CPU_CONTROLLER_inst : CPU_CONTROLLERV3
--			Generic map(
----				NBRWINDOWS => 256
--			)
			Port map(
			nrst			=> nrst,
			ClockBus		=> ClockBus,
			--timecounter		=> timecounter,
			timestamp  => timestamp,
			trigger 	=> trigger,

			CPUBus		=> Bus_intl,
			CPUTime		=> Time_intl,
			TriggerInfo	=> TrigInfo_intl,

			WR_RS_S			=> WR_RS_S_user,
			WR_CS_S			=> WR_CS_S_user,

			CtrlBus_IxSL => CtrlBus_IxSL,
			ValidReal 	=> ValidReal_s,
			ValidData	=> ValidData_s,
			--CurAddr			=> CurWindowCnt,
		--	RealAddrBit		=> RealAddrBit_s,
			--OldAddr			=> OldWindowCnt,
			--OldAddrBit		=> OldAddrBit_s,

			-- NextAddr_in		=> NextAddrBus,
			--PrevAddr_in		=> PrevAddrBus,
			NextAddr_in		=> NextAddr_s,
			PrevAddr_in		=> PrevAddr_s,
			NextValid_in	=> NextValid_s,
			PrevValid_in	=> PrevValid_s,

			-- FIFO IN for Digiting
			DIG_Full		=> DIG_Full,
			DIG_DataIn		=> DIG_DataIn,
			DIG_WriteEn		=> DIG_WriteEn
			);




	WDOSTORE : WindowStoreV4
--		Generic map(
----		NBRWINDOWS => NBRWINDOWS
--		)
		Port map(
		nrst			=> nrst,
		ClockBus	=> ClockBus,
		ValidData	=> ValidData_s,

		CPUBus		=> Bus_intl,
		CPUTime		=> Time_intl,
		TriggerInfo	=> TrigInfo_intl,
		trigger=>       trigger,

		  -- Control Signals
        CtrlBus_IxSL => CtrlBus_IxSL ,

		-- Overwatch
		--NbrOfPackets	=> CtrlBus_OxSL.RBNbrOfPackets,
		NbrOfPackets	=> RBNbrOfPackets,
		Reg_Clr			=> CtrlBus_IxSL.WindowStorage,

		-- FIFO out for Reading RDAD
		RDAD_ReadEn  => RDAD_ReadEn,
		RDAD_DataOut => RDAD_DataOut,
		RDAD_Empty	=> 	RDAD_Empty,
        RDAD_Data_trig => RDAD_Data_s,
        RDAD_WriteEn_trig =>RDAD_WrEn_s,
        
        
        
		AXI_ReadEn			=> AXI_ReadEn,
		AXI_Time_DataOut	=> AXI_Time_DataOut,
		AXI_WdoAddr_DataOut	=> AXI_WdoAddr_DataOut,
		AXI_TrigInfo_DataOut	=> AXI_TrigInfo_DataOut,
		AXI_Spare_DataOut 		=> AXI_Spare_DataOut,
		AXI_Empty	=> AXI_Empty
		);

 nextAddressCnt_inst : nextAddressCnt
        Port map(
        clk => ClockBus.CLK125MHz,
        rst => nrst,   
        usrRst => CtrlBus_IxSL.WindowStorage,                   
        Ctrl => ValidReal_s,
        nextAddress => NextAddr_s,
        nextAddressValid => NextValid_s,
        address_is_zero => address_is_zero_out

        ); 



	
multiplex_WR:	process(ClockBus.CLK125MHz, nrst)
        begin
            if nrst = '0' then
              WR_RS_S  <=  (others => '0');
              WR_CS_S  <=  (others => '0');
           
            
            else
                if rising_edge(ClockBus.Clk125MHz) then
                    if CtrlBus_IxSL.CPUMode = '0' then
                         WR_RS_S  <=  WR_RS_S_user;
                         WR_CS_S <=  WR_CS_S_user; 
                    elsif CtrlBus_IxSL.CPUMode = '1' then
                        WR_RS_S  <=  WR_RS_S_trig;
                        WR_CS_S <=  WR_CS_S_trig; 
                    else
                        WR_RS_S  <=  (others => '0');
                        WR_CS_S  <=  (others => '0');
                         
	                end if;
	            end if;    
	        end if;
	end process;



trigger_s <= '0' when trigger= "0000" else '1';



end implementation;
