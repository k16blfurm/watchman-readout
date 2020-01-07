
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.TARGETC_pkg.all;
use work.WindowCPU_pkg.all;

entity WindowStoreV4 is
--	Generic(
--		NBRWINDOWS : integer := 16
--	);
	Port (
	nrst : 			in	std_Logic;
	ClockBus:		in T_ClockBus;

	ValidData:		in	std_logic;

	CPUBus:			in 	std_logic_vector(10 downto 0);
	CPUTime:		in	T_timestamp;
	TriggerInfo:	in 	std_logic_vector(11 downto 0);
	trigger:        in  std_logic_vector(3 downto 0);

	
    -- Control Signals
    CtrlBus_IxSL:    in     T_CtrlBus_IxSL;

	-- Overwatch of Transmission
	NbrOfPackets:	out	std_logic_vector(7 downto 0);
	Reg_Clr:		in	std_logic;

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

end WindowStoreV4;

architecture Behavioral of WindowStoreV4 is
--	component aFifoV2 is
--    generic (
--        DATA_WIDTH :integer := 8;
--        ADDR_WIDTH :integer := 4
--    );
--    port (
--    	rst :		in std_logic;
--        -- Reading port.
--        Data_out    :out std_logic_vector (DATA_WIDTH-1 downto 0);
--        Empty_out   :out std_logic;
--        ReadEn_in   :in  std_logic;
--        RClk        :in  std_logic;
--        -- Writing port.
--        Data_in     :in  std_logic_vector (DATA_WIDTH-1 downto 0);
--        Full_out    :out std_logic;
--        WriteEn_in  :in  std_logic;
--        WClk        :in  std_logic
--    );
--	end component aFifoV2;

COMPONENT axi_wdo_addr_fifo
  PORT (
   -- rst : IN STD_LOGIC;
    wr_clk : IN STD_LOGIC;
    rd_clk : IN STD_LOGIC;
    din : IN STD_LOGIC_VECTOR(8 DOWNTO 0);
    wr_en : IN STD_LOGIC;
    rd_en : IN STD_LOGIC;
    dout : OUT STD_LOGIC_VECTOR(8 DOWNTO 0);
    full : OUT STD_LOGIC;
    empty : OUT STD_LOGIC
  );
END COMPONENT;


COMPONENT axi_cmd_fifo_11W_5D
  PORT (
    wr_clk : IN STD_LOGIC;
    rd_clk : IN STD_LOGIC;
    din : IN STD_LOGIC_VECTOR(10 DOWNTO 0);
    wr_en : IN STD_LOGIC;
    rd_en : IN STD_LOGIC;
    dout : OUT STD_LOGIC_VECTOR(10 DOWNTO 0);
    full : OUT STD_LOGIC;
    empty : OUT STD_LOGIC
  );
END COMPONENT;

COMPONENT axi_time_fifo_64W_32D
  PORT (
    wr_clk : IN STD_LOGIC;
    rd_clk : IN STD_LOGIC;
    din : IN STD_LOGIC_VECTOR(63 DOWNTO 0);
    wr_en : IN STD_LOGIC;
    rd_en : IN STD_LOGIC;
    dout : OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    full : OUT STD_LOGIC;
    empty : OUT STD_LOGIC
  );
END COMPONENT;
COMPONENT axi_trig_afifo_12W_32D
  PORT (
    wr_clk : IN STD_LOGIC;
    rd_clk : IN STD_LOGIC;
    din : IN STD_LOGIC_VECTOR(11 DOWNTO 0);
    wr_en : IN STD_LOGIC;
    rd_en : IN STD_LOGIC;
    dout : OUT STD_LOGIC_VECTOR(11 DOWNTO 0);
    full : OUT STD_LOGIC;
    empty : OUT STD_LOGIC
  );
END COMPONENT;
        
--COMPONENT trig_fifo_3W_16D
--  PORT (
--    wr_clk : IN STD_LOGIC;
--    rd_clk : IN STD_LOGIC;
--    din : IN STD_LOGIC_VECTOR(3 DOWNTO 0);
--    wr_en : IN STD_LOGIC;
--    rd_en : IN STD_LOGIC;
--    dout : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
--    full : OUT STD_LOGIC;
--    empty : OUT STD_LOGIC
--  );
--END COMPONENT;


	type T_storestate is(
		IDLE,
		STABILIZE,
		PREPARE,
		PULSE,
		STABILIZE2,
		PREPARE2,
		PULSE2
	);

	constant TRIG_LAST_MASK : std_logic_vector(11 downto 0) := "000000001111";

	signal writeEn_stm : T_storestate := IDLE;

	signal Full_out_intl    : std_logic;
	signal WriteEn_intl  : std_logic;
	signal WriteEn  : std_logic;

	signal Counter:		std_logic_vector(63 downto 0);

	--signal TrigInfoDly, TrigInfoBuf, TrigInfoBuf_dly : std_logic_vector(11 downto 0);

	signal Trig:	std_logic_vector(11 downto 0);

	signal Wdo1:	std_logic_vector(8 downto 0);
    signal WdoNumber:	std_logic_vector(8 downto 0);

	signal CMD_s:	std_logic_vector(10 downto 0);
    signal CMD_s1:	std_logic_vector(10 downto 0);

	signal axi_full_s:	std_logic_vector(3 downto 0);
	signal axi_empty_s:	std_logic_vector(3 downto 0);

	signal NbrOfPackets_intl : std_logic_vector(7 downto 0);
	   signal trigger125MHz_s:     std_logic_vector(3 downto 0);
     signal trigger_empty_s:        std_logic;
    signal trigger_full_s:        std_logic;
  
	-- -------------------------------------------------------------
	-- Constraints on Signals
	-- -------------------------------------------------------------
	--attribute DONT_TOUCH : string;
	--attribute DONT_TOUCH of RDAD_STO_AFIFO : label is "TRUE";
	--attribute DONT_TOUCH of writeEn_stm : signal is "TRUE";
	--attribute DONT_TOUCH of writeen_intl : signal is "TRUE";

begin


	process(ClockBus.CLK125MHz)
	begin
		if nrst = '0' then
			--RDAD_WriteEn_intl <= '0';
			--AXI_WriteEn_intl <= '0';
			NbrOfPackets_intl <= (others => '0');
			trig <= (others => '0');
			Wdo1 <= (others => '0');
			counter <= (others => '0');
			cmd_s1 <= (others => '0');
		else
			if rising_edge(ClockBus.Clk125MHz) then

				if Reg_CLR = '1' then
					NbrOfPackets_intl <= (others => '0');
				end if;

				if ValidData = '1' then
					case writeEn_stm is
						when IDLE =>
							case CPUBus(10 downto 8) is
								when CMD_WR1_MARKED => -- ODD WINDOW
									NbrOfPackets_intl <= std_logic_vector(unsigned(NbrOfPackets_intl)+1);
								--	counter <= CPUTime.graycnt & "0000"; -- gray counter is a timestamp for the window
									--trig <= TrigInfoDly;
								--	trig <= TriggerInfo;
									Wdo1 <= CPUBus(7 downto 0) & '0';
									cmd_s1 <= CPUBus;
									writeEn_stm <= STABILIZE;
								when CMD_WR2_MARKED =>-- EVEN WINDOW
									NbrOfPackets_intl <= std_logic_vector(unsigned(NbrOfPackets_intl)+1);
									--counter <= CPUTime;
									--counter <= CPUTime.graycnt & "1000";
									--trig <= TrigInfoDly;
								--	trig <= TriggerInfo;
									Wdo1 <= CPUBus(7 downto 0) & '1';
									cmd_s1 <= CPUBus;
									writeEn_stm <= STABILIZE;
								when CMD_BOTH_MARKED =>
									NbrOfPackets_intl <= std_logic_vector(unsigned(NbrOfPackets_intl)+2);
							--		counter <= CPUTime.graycnt & "0000";
									--trig <= TrigInfoDly;
								--	trig <= TriggerInfo and TRIG_LAST_MASK;
									Wdo1 <= CPUBus(7 downto 0) & '0';
									cmd_s1 <= CPUBus;
									writeEn_stm <= STABILIZE;
								when others =>
								writeEn_stm <= IDLE;
							end case;



						when STABILIZE =>
							writeEn_stm <= PREPARE;
						when PREPARE =>
							writeEn_stm <= PULSE;

						when PULSE =>
							if cmd_s1(10 downto 8) = CMD_BOTH_MARKED then
								Wdo1 <= Wdo1(8 downto 1) & '1';
								--trig <= TriggerInfo;
						--		Counter <= Counter(63 downto 4) & "1000";
								writeEn_stm <= STABILIZE2;
							else
								writeEn_stm <= IDLE;
							end if;
						when STABILIZE2 =>

							writeEn_stm <= PREPARE2;
						when PREPARE2 =>
							--RDAD_WriteEn_intl <= '0';
							--AXI_WriteEn_intl <= '1';
							writeEn_stm <= PULSE2;
						when PULSE2 =>
							--RDAD_WriteEn_intl <= '0';
							--AXI_WriteEn_intl <= '0';
							writeEn_stm <= IDLE;

						when others =>
							--RDAD_WriteEn_intl <= '0';
							--AXI_WriteEn_intl <= '0';
							writeEn_stm <= IDLE;
					end case;
				end if;
			end if;
		end if;
	end process;

	process(ClockBus.CLK125MHz)
	begin
		if nrst = '0' then
			WriteEn_intl <= '0';
		else
			if rising_edge(ClockBus.Clk125MHz) then

				if ValidData = '1' then
					case writeEn_stm is
						when STABILIZE =>
							WriteEn_intl <= '1';
						when STABILIZE2 =>
							WriteEn_intl <= '1';
						-- when PREPARE =>
						-- 	WriteEn_intl <= '1';
						-- when PREPARE2 =>
						-- 	WriteEn_intl <= '1';
						when others =>
							WriteEn_intl <= '0';
					end case;
				else
					WriteEn_intl <= '0';
				end if;
			end if;
		end if;
	end process;
	
	

	
	
	
multiplex_WdoNumber:	process(ClockBus.CLK125MHz)
        begin
            if nrst = '0' then
                WriteEn <= '0';
                WdoNumber <= (others=>'0'); 
            else
                if rising_edge(ClockBus.Clk125MHz) then
                    if CtrlBus_IxSL.CPUMode = '0' then
                        WriteEn <= WriteEn_intl;
                        WdoNumber <= Wdo1; 
                    else
                         WriteEn <=RDAD_WriteEn_trig ;
                         WdoNumber <= RDAD_Data_trig; 
	                end if;
	            end if;    
	        end if;
	end process;


		
    Cmd_s<= (others => '0');
    Trig <= (others => '0');
    counter <= (others => '0');
-- Window address to RDAD_ADD module

    RDAD_STO_AFIFO : axi_wdo_addr_fifo
      PORT MAP (
      --  rst => nrst,
        
        dout => RDAD_DataOut,
        empty => RDAD_Empty,
        rd_en => RDAD_ReadEn,
        rd_clk => ClockBus.RDAD_CLK,

        
        din => WdoNumber,
        full => Full_out_intl,
        wr_en => WriteEn,
        wr_clk => ClockBus.CLK125MHz

      );
    

-- Window address for DIG time in FIFO manager
    
    
    AXI_CMD_AFIFO :  axi_cmd_fifo_11W_5D
    port map (

     dout => AXI_Spare_DataOut, 
     empty => axi_empty_s(0),
     rd_en => AXI_ReadEn,
     rd_clk => ClockBus.AXI_CLK,

     
     din => Cmd_s,
     full => axi_full_s(0),
     wr_en => WriteEn,
     wr_clk => ClockBus.CLK125MHz
     );
     
 
 
  -- Counter for WDOTime in FIFO manager

    
    AXI_Time_AFIFO :  axi_time_fifo_64W_32D
    port map (

     dout => AXI_Time_DataOut,
     empty => axi_empty_s(1),
     rd_en => AXI_ReadEn,
     rd_clk => ClockBus.AXI_CLK,

     
     din => Counter,
     full => axi_full_s(1),
     wr_en => WriteEn,
     wr_clk => ClockBus.CLK125MHz
     );
     
    

 -- FIFO_WdoAddr for  FIFO manager


    AXI_WdoAddr_AFIFO : axi_wdo_addr_fifo
      PORT MAP (
      --  rst => nrst,
        
        dout => AXI_WdoAddr_DataOut,
        empty => axi_empty_s(2),
        rd_en => AXI_ReadEn,
        rd_clk => ClockBus.AXI_CLK,

        
        din => WdoNumber,
        full => axi_full_s(2),
        wr_en => WriteEn,
        wr_clk => ClockBus.CLK125MHz

      );
    
    
  

-- Trigger info  for FIFO_TrigInfo in FIFO MANAGER
    
    AXI_Trig_AFIFO :  axi_trig_afifo_12W_32D
    port map (

     dout => AXI_TrigInfo_DataOut,
     empty => axi_empty_s(3),
     rd_en => AXI_ReadEn,
     rd_clk => ClockBus.AXI_CLK,

     
     din => Trig,
     full => axi_full_s(3),
     wr_en => WriteEn,
     wr_clk => ClockBus.CLK125MHz
     );
     



	AXI_empty <= '0' when axi_empty_s = "0000" else '1';
	NbrOfPackets <= NbrOfPackets_intl;
	
	
	
	
--        trigger_100to125MHz : trig_fifo_3W_16D
--        port map (
    
--         dout => trigger125MHz_s,
--         empty => trigger_empty_s,
--         rd_en => AXI_ReadEn,
--         rd_clk => ClockBus.CLK125MHz,
    
         
--         din => trigger,
--         full => trigger_full_s,
--         wr_en => CtrlBus_IxSL.WindowStorage,
--         wr_clk => ClockBus.AXI_CLK
--         );
         

	
	
end Behavioral;