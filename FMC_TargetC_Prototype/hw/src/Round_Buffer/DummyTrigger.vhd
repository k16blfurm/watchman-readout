----------------------------------------------------------------------------------
-- Company: IDLAB, Hawaii
-- Engineer: Salvador Ventura
-- 
-- Create Date: 09/11/2019
-- Design Name: 
-- Module Name: DummyTrigger - 
-- Project Name: WATCHMAN
-- Target Devices: 
-- Tool Versions: 

-- Description: A strobe simulating trigger for testing circular buffer.
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
 use ieee.numeric_std.all;
 use work.TARGETC_pkg.all;
entity DummyTrigger is

 
    port (

  clk :            in  std_logic;
  RST :             in  std_logic;
  start_reg:	    in 	std_logic;
  Timestamp:        in T_timestamp;
 
  trigger :         out std_logic

);

end DummyTrigger;
 
architecture structure of DummyTrigger is

signal trigger_i : std_logic;

signal cnt_i : std_logic_vector(8 downto 0);


attribute mark_debug : string;

attribute mark_debug of cnt_i: signal is "true";

type T_stm_dummyTrigger is(
IDLE,
Running
	);

signal stm_dummyTrigger: T_stm_dummyTrigger;

begin

p_trigger: process(clk, RST, start_reg, Timestamp.samplecnt)

begin


if RST = '0' or start_reg='0' then

   trigger_i <= '0';
   cnt_i <= (others => '0');
   stm_dummyTrigger <= IDLE;

else
   if rising_edge(clk) then
        case stm_dummyTrigger is
          when IDLE =>
               if (start_reg = '1') and (Timestamp.samplecnt="111") then
                   stm_dummyTrigger <= Running;
               else
                   stm_dummyTrigger<= IDLE;
               end if;
              
          when Running =>
          --    if rising_edge(clk) then
                if (start_reg = '1') then
                --	cnt_i <= (others => '0');
                    if cnt_i < "000001000" then
                        cnt_i <= std_logic_vector(unsigned(cnt_i) + 1);
                        trigger_i <= '0';
                        stm_dummyTrigger <= Running;

                    else
                        if cnt_i < "000001011" then
                            cnt_i <= std_logic_vector(unsigned(cnt_i) + 1);
                            trigger_i <= '1';
                            stm_dummyTrigger <= Running;

                        else
                            cnt_i <= (others => '1');
                            trigger_i <= '0';
                            stm_dummyTrigger <= Running;

                        end if;
                    end if;
                else
                    
                    cnt_i <= (others => '0');
                    stm_dummyTrigger <= Running;

                end if;
                stm_dummyTrigger <= Running;
           
 --             end if;
          end case;
        
   end if;
end if;
end process;

trigger<= trigger_i;

end architecture;






