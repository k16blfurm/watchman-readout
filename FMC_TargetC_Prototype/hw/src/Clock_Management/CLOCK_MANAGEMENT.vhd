
-- Clock Wizard for :
		-- |	Name	|	Period	|	Frequency	|
		--  ------------|-----------|---------------|
		-- |	SSTIN	|	64 ns	|	15.625 MHz	|
		-- |	SCLK	|	...		|	250 MHz		|	Derived from SSTIN main clock 16*SSTIN
		-- |	RDAD_CLK|	...		|	250 MHz		|	Derived from SSTIN main clock 16*SSTIN
		-- |	HSCLK	|	...		|	250 MHz		|	Derived from SSTIN main clock 16*SSTIN
		-- |	WL_CLK	|	...		|	MAXFREQ		|	Derived from SSTIN main clock but Variable [SSTIN:MAXFREQ]


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.TARGETC_pkg.all;

library unisim;
use unisim.vcomponents.all;

entity TC_ClockManagementV3 is

	port (
	-- TARGET C Ports for control and function
		nrst:			in	std_logic;
		clk1:			in 	std_logic;	-- Clock for the TARGETC SCLK
		clk2:			in 	std_logic;	-- Clock for the TARGETC SCLK
		AXI_Clk:		in	std_logic;

		WL_CLK_DIV:		in 	std_logic_vector(31 downto 0); -- Clock Divider Through DFF

		PLL_LOCKED:		out	std_logic;

		ClockBus:		out T_ClockBus;

		Timecounter:	out std_logic_vector(63 downto 0);
		Timestamp:		out T_timestamp;

		HSCLKdif:		in std_logic;		-- Pin#43 to Pin#44

		-- LVDS Differential Pair
		HSCLK_P:		out std_logic;		-- Pin#43
		HSCLK_N:		out std_logic;		-- Pin#44

		WL_CLK_P:		out std_logic;		-- Pin#57
		WL_CLK_N:		out std_logic;		-- Pin#58

		SSTIN_P:		out std_logic;		-- Pin#125
		SSTIN_N:		out std_logic		-- Pin#126

	);
end TC_ClockManagementV3;

architecture arch_imp of TC_ClockManagementV3 is

	-- --------------------------------------------------------------------------------
	-- Component Declaration
	component OBUFDS is
	generic(
		CAPACITANCE : string     := "DONT_CARE";
		IOSTANDARD  : string     := "DEFAULT"
	);

	port(
		O  : out std_ulogic;
		OB : out std_ulogic;

		I : in std_ulogic
	);
	end component OBUFDS;

	component counter is
	generic(
		NBITS : integer := 8
	);
	port(
		CLK : in  std_logic;
		RST : in  std_logic;
		Q : out std_logic_vector(NBITS-1 downto 0)
	);
	end component counter;

	component GRAY_ENCODER is
		generic(
			NBITS : integer := 8
		);
		Port (
		GRAY_OUT :	out 	std_logic_vector(NBITS-1 downto 0);
		BIN_IN:		in	std_logic_vector(NBITS-1 downto 0)
		);
	end component GRAY_ENCODER;
	
	
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
	
	
	-- --------------------------------------------------------------------------------
	-- Signal Declaration

	signal locked_WL:	std_logic;
	signal locked_general,locked_general2, locked_general_sync125MHz, locked_general_sync100MHz:	std_logic;
	signal locked_intl, locked_intl125MHz, locked_intl100MHz:	std_logic;
    signal nrst_125MHz: std_logic;
	-- Internal Clock signals (intl)

	signal SCLKraw :		std_logic;
	signal HSCLKraw :		std_logic;
	signal RDAD_CLKraw :	std_logic;
	--signal CLK500MHzraw :	std_logic;
	--signal CLK500MHzraw2 :	std_logic;
	signal CLK100MHzraw :	std_logic;
	signal CLK150MHzraw :	std_logic;
	signal CLK200MHzraw :	std_logic;
	signal CLK125MHzraw :	std_logic;

	signal sstin_bufg : std_logic;
	signal CLK100MHz_bufg :	std_logic;
	signal CLK125MHz_bufg :	std_logic;


	signal SSTIN_intl:		std_logic;
	signal SCLK_intl :		std_logic;
	signal HSCLK_intl :		std_logic;
	signal RDAD_CLK_intl :	std_logic;
--	signal CLK500MHz_intl :	std_logic;
	signal CLK125MHz_intl :	std_logic;
	signal WL_CLK_intl :		std_logic;


	signal WL_CLK_100MHz_bufg : std_logic;
	signal WL_CLK_150MHz_bufg : std_logic;
	signal WL_CLK_200MHz_bufg : std_logic;
	signal WL_CLK_125MHz_bufg : std_logic;
	signal WL_CLK_300MHz_bufg : std_logic;

	signal rst :		std_logic;
	signal Timecounter_intl : std_logic_vector(63 downto 0);
	signal GrayCounter_intl : std_logic_vector(59 downto 0);
	signal clkFbOut_WL, clkFbOut : std_logic;
	signal clkFbIn_WL, clkFbIn : std_logic;
	signal axi_clk_intl:std_logic;

begin

	rst <= not nrst;

	  -- MMCM to generate base and fast clocks
	  -- fout = Fclkin x M/(D*O)
--	TC_CLK_MNG_inst : MMCME2_ADV
--	generic map(
--		BANDWIDTH			=> "OPTIMIZED",
--		COMPENSATION         => "BUF_IN",
--		STARTUP_WAIT         => false,
--		DIVCLK_DIVIDE        => 1,     -- D
--		CLKFBOUT_MULT_F      => 1.0,  -- M 10.0 for 125 mhz from 100 mhz
--		CLKFBOUT_PHASE       => 0.000,
--		CLKFBOUT_USE_FINE_PS => false,
--		CLKOUT0_DIVIDE_F       => 1.0, -- 169  M CLKO
--		CLKOUT0_PHASE        => 0.000,
--		CLKOUT0_DUTY_CYCLE   => 0.500,
--		CLKOUT0_USE_FINE_PS  => false,
--		--
--		CLKOUT1_DIVIDE      => 1, -- 169   -- O  changed from 4 to 10 to get 100MHz, 8 to get 125
--		CLKOUT1_PHASE        => 0.000,
--		CLKOUT1_DUTY_CYCLE   => 0.500,
--		CLKOUT1_USE_FINE_PS  => false,
--		--
--		CLKIN1_PERIOD        => 8.0,
--		--CLKIN2_PERIOD        => 10.0,
--		REF_JITTER1          => 0.010,
--		REF_JITTER2          => 0.010)
--	port map(
--		-- Output clocks
--		CLKFBOUT     => clkFbOut,
--		CLKFBOUTB    => open,
		
--		CLKOUT0      => Clk100MHzRaw,
--		CLKOUT0B     => open,
--		CLKOUT1      => CLK125MHzRaw,
		
--		CLKOUT1B     => open,
--		CLKOUT2      => open,
--		CLKOUT2B     => open,
--		CLKOUT3      => open,
--		CLKOUT3B     => open,
--		CLKOUT4      => open,
--		CLKOUT5      => open,
--		CLKOUT6      => open,
--		-- Input clock control
--		CLKFBIN      => clkFbIn,
--		CLKIN1       => clk1,
--		CLKIN2       => '0',
--		-- Tied to always select the primary input clock
--		CLKINSEL     => '1',
--		-- Ports for dynamic reconfiguration
--		DADDR        => (others => '0'),
--		DCLK         => '0',
--		DEN          => '0',
--		DI           => (others => '0'),
--		DO           => open,
--		DRDY         => open,
--		DWE          => '0',
--		-- Ports for dynamic phase shift
--		PSCLK        => '0',
--		PSEN         => '0',
--		PSINCDEC     => '0',
--		PSDONE       => open,
--		-- Other control and status signals
--		LOCKED       => locked_general,
--		CLKINSTOPPED => open,
--		CLKFBSTOPPED => open,
--		PWRDWN       => '0',
--		RST          => rst
--	);

--   BUFG_FB : BUFG
--      port map (
--         I => clkFbOut,
--         O => clkFbIn
--   );

--	BUFG_OUTCLK100 : BUFG
--	port map (
--    	I => Clk100MHzRaw,
--    	O => Clk100MHz_bufg
--	);

--	BUFG_OUTCLK150 : BUFG
--	port map (
--    	I => CLK125MHzRaw,
--    	O => CLK125MHz_bufg
--	);

--SyncBitLocked125MHz: SyncBit
--       generic map (
--          SYNC_STAGES_G  => 2,
--          CLK_POL_G      => '1',
--          RST_POL_G      => '1',
--          INIT_STATE_G   => '0',
--          GATE_DELAY_G   => 1 ns
--       )
       
--       port map ( 
--          -- Clock and reset
--          clk  => CLK125MHz_bufg,
--          rst   => rst,
--          -- Incoming bit, asynchronous
--          asyncBit =>  locked_general,
--          -- Outgoing bit, synced to clk
--          syncBit   => locked_general_sync125MHz
--       ); 
	
	
--	SyncBitLocked100MHz: SyncBit
--              generic map (
--                 SYNC_STAGES_G  => 2,
--                 CLK_POL_G      => '1',
--                 RST_POL_G      => '1',
--                 INIT_STATE_G   => '0',
--                 GATE_DELAY_G   => 1 ns
--              )
              
--              port map ( 
--                 -- Clock and reset
--                 clk  => Clk100MHz_bufg,
--                 rst   => rst,
--                 -- Incoming bit, asynchronous
--                 asyncBit =>  locked_general,
--                 -- Outgoing bit, synced to clk
--                 syncBit   => locked_general_sync100MHz
--              ); 
           

	locked_intl <= 	'1';
	--CLK500MHz_intl <= '0' when locked_intl = '0' else CLK500MHzraw;
--	CLK125MHz_intl <= '0' when locked_intl = '0' else CLK125MHz_bufg;
	

--	SSTIN_intl <= '0' when locked_intl = '0' else SSTIN_bufg;


--	RDAD_CLK_intl <= '0' when locked_intl = '0' else Clk100MHzRaw;

--	SCLK_intl <= '0' when locked_intl = '0' else Clk100MHzRaw;

--	HSCLK_intl <= '0' when locked_intl = '0' else Clk100MHzRaw;

--	WL_CLK_intl <= '0' when locked_intl = '0' else Clk100MHzRaw;

    --axi_clk_intl <= '0' when locked_intl = '0' else Clk100MHzRaw;




--	locked_intl125MHz <= 	'0' when locked_general_sync125MHz = '0' else '1';
--	locked_intl100MHz <= 	'0' when locked_general_sync100MHz = '0' else '1';


--	--CLK500MHz_intl <= '0' when locked_intl = '0' else CLK500MHzraw;
--	CLK125MHz_intl <= '0' when locked_intl125MHz = '0' else CLK125MHz_bufg;

--	RDAD_CLK_intl <= '0' when locked_intl100MHz = '0' else Clk100MHz_bufg;

--	SCLK_intl <= '0' when locked_intl100MHz = '0' else Clk100MHz_bufg;

--	HSCLK_intl <= '0' when locked_intl100MHz = '0' else Clk100MHz_bufg;

--	WL_CLK_intl <= '0' when locked_intl100MHz = '0' else Clk100MHz_bufg;

--	SSTIN_intl <= '0' when locked_intl100MHz = '0' else SSTIN_bufg;

	--	------------------------------------------------------------
	-- Output For Standard and LVDS

    
--SyncBit_reset125MHz: SyncBit
--       generic map (
--          SYNC_STAGES_G  => 2,
--          CLK_POL_G      => '1',
--          RST_POL_G      => '1',
--          INIT_STATE_G   => '0',
--          GATE_DELAY_G   => 1 ns
--       )
       
--       port map ( 
--          -- Clock and reset
--          clk  => CLK125MHz_bufg,
--          rst   => '0',
--          -- Incoming bit, asynchronous
--          asyncBit =>  nrst,
--          -- Outgoing bit, synced to clk
--          syncBit   => nrst_125MHz
--       ); 
	

	TimeCnt_inst : counter
	generic map(
		NBITS => 64
	)
	port map(
		CLK 	=> AXI_CLK,
		RST => nrst, 
		Q 	=> Timecounter_intl
	);

	GrayTime :  GRAY_ENCODER
		generic map(
			NBITS => 60
		)
		Port map(
			GRAY_OUT => GrayCounter_intl,
			BIN_IN	=> Timecounter_intl(63 downto 4)
		);

	-- Timestamp
	Timestamp.graycnt <= GrayCounter_intl;
	Timestamp.samplecnt <= Timecounter_intl(2 downto 0);

	TimeCounter <= Timecounter_intl;


	SSTIN_intl	<= not Timecounter_intl(2);

	OBUFDF_SSTIN : OBUFDS
	generic map(
		IOSTANDARD  => "BLVDS_25"
	)
	port map(
		O	=> SSTIN_P,
		OB	=> SSTIN_N,

		I	=> SSTIN_intl
	);
	


	OBUFDF_WL_CLK : OBUFDS
	generic map(
		IOSTANDARD  => "LVDS_25"
	)
	port map(
		O	=> WL_CLK_P,
		OB	=> WL_CLK_N,

		I	=> AXI_CLK
	);

	OBUFDF_HSCLK : OBUFDS
	generic map(
		IOSTANDARD  => "LVDS_25"
	)
	port map(
		O	=> HSCLK_P,
		OB	=> HSCLK_N,

		I	=> HSCLKdif
	);
	
	
	-- CLOCK BUS OUTPUTS
	
    
        ClockBus.SCLK     <= AXI_CLK;
        ClockBus.HSCLK    <= AXI_CLK;
        ClockBus.WL_CLK    <= AXI_CLK;
        ClockBus.RDAD_CLK<= AXI_CLK;
        
        ClockBus.CLK125MHz    <= AXI_CLK;
        ClockBus.SSTIN    <= SSTIN_intl;
        ClockBus.AXI_CLK <= AXI_CLK;

	
		PLL_LOCKED <= locked_intl; -- signal to use in TARGETC_Control.vhd for registers


end arch_imp;
