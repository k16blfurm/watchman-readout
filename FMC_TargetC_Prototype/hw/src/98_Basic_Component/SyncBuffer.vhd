library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity SyncBuffer is
    generic(
	NBITS : integer := 32
    );
    port (
          -- Clock and reset
	Clk:	in  std_logic;
	nrst:	in  std_logic;
      -- Incoming buffer, asynchronous
	asyncBuffer:	in  std_logic_vector(NBITS-1 downto 0);
      -- Outgoing buffer, synced to clk
	syncBuffer:     out std_logic_vector(NBITS-1 downto 0)
--	ClkA:	in  std_logic;
    );
end SyncBUffer;

architecture rtl of SyncBuffer is

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

    -- ----------------------------------
    -- Signals

    signal syncBuffer_intl : std_logic_vector(NBITS-1 downto 0);
--  signal B_intl : std_logic_vector(NBITS-1 downto 0);
--  signal QB_intl : std_logic_vector(NBITS-1 downto 0);

    -- -------------------------------------------------------------
    -- Constraints on Signals
    -- -------------------------------------------------------------
    attribute DONT_TOUCH : string;
    attribute DONT_TOUCH of syncBuffer_intl: signal is "TRUE";
--  attribute DONT_TOUCH of B_intl: signal is "TRUE";
--  attribute DONT_TOUCH of QB_intl: signal is "TRUE";

begin

    SyncBuffer_GEN : for I in 0 to NBITS-1 generate
	DFF : SyncBit
       generic map (
              SYNC_STAGES_G  => 2,
              CLK_POL_G      => '1',
              RST_POL_G      => '1',
              INIT_STATE_G   => '0',
              GATE_DELAY_G   => 1 ns
           )
           
           port map ( 
              -- Clock and reset
              clk  => clk,
              rst   => nrst,
              -- Incoming buffer bit, asynchronous
              asyncBit => asyncBuffer(I),
              -- Outgoing buffer bit, synced to clk
              syncBit   => syncBuffer(I)
           ); 

    end generate;

    syncBuffer <= syncBuffer_intl;
end rtl;

