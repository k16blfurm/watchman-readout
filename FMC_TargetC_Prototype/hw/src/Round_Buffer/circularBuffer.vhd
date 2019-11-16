----------------------------------------------------------------------------------
-- Company: IDLAB, Hawaii
-- Engineer: Salvador Ventura
-- 
-- Create Date: 09/11/2019
-- Design Name: 
-- Module Name: circularBuffer - 
-- Project Name: WATCHMAN
-- Target Devices: 
-- Tool Versions: 

-- Description: This module controls the writting process in TargetC in function of 
-- a trigger. The main part is done by a state machine with two types of states:
-- hit and wr_add. All the states monitor the trigger signal but the wr_add also
-- contains the logic for moving to the next address. The addresses corresponds to 
-- the number of windows considering 2-windows subbuffers.
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
 
entity circular_buffer is

  port (
          clk : in  std_logic;
          RST : in  std_logic;  
          trigger : in std_logic;
          full_fifo : in std_logic;          
          ptr_window : out std_logic_vector(8 downto 0);
          sstin :  out std_logic;
       --   wr:     out unsigned(8 downto 0);
          enable_write :  out std_logic;
          counter :  out std_logic_vector(2 downto 0);
          
          RD_add: out std_logic_vector(8 downto 0);
          WR_RS: out std_logic_vector(1 downto 0);
          WR_CS: out std_logic_vector(5 downto 0)
  );
end circular_buffer;
 
architecture structure of circular_buffer is
 
signal ptr_window_i : std_logic_vector(8 downto 0);
signal counter_i: std_logic_vector(2 downto 0);
signal sstin_i : std_logic;
signal flag, flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8, flag9, flag10, flag11, flag12, flag13, flag14, flag15 : boolean;
signal flag_full: std_logic;
signal window2read: std_logic_vector(8 downto 0);
signal wr_shifted: unsigned(8 downto 0);
signal ptr_window_trans_i: std_logic_vector(8 downto 0);
--signal ptr_sub_i: unsigned(8 downto 0);

type stmachine is ( wr_add ,hit, hit2, hit3, hit4, hit5, hit6,hit7,hit8, hit9, hit10, hit11, hit12, hit13,hit14, hit15);
signal stm: stmachine;


type stmachine_comp is ( A, B, C, D);


signal stm_comp: stmachine_comp;
signal saved_i: std_logic_vector(8 downto 0);
signal enable_write_i: std_logic;
--signal rd_add_i : std_logic_vector(8 downto 0);
--signal flag_no_hit_last_state: boolean;

  begin
  
  ----------------------------------
  -- State machine for handling the trigger and to generate the 
  -- wr/read addresses (signals WR_CS, WR_RS, RD_add and enable_write). 
  
  -- The ptr_window_i counter is running over the number of winwows 
  -- increasing +4 (for a 2-write-address subbuffer).
  
  -- Signal wr_shifted is ptr_window divided by two (one-bit shift)
  -- to get the real address WR_CS and WR_RS for the TARGETC. This operation
  -- is done in the first state, then, the wr_shifted is just increased by 1.

  
  -- Signal ptr_sub_i is the current subbuffer window. Uncomment lines were ptr_sub_i appeared
  -- to follow the signal 
  
  -- The first 15 states are for raising flags and to get wr from
  -- the pointer_window_i signal 
  
  
  -- window2read could be modified to get the right window according to the trigger delay
  ----------------------------------
 
  p_sm:  process(clk,RST, trigger, full_fifo)
  begin 
 if RST = '0' then
      stm <= hit;
      ptr_window_i  <= (others=> '0');
--      ptr_sub_i  <= (others=> '0');

      flag_full <= '0';
      window2read <= (others=> '0');
      wr_shifted <= (others=> '0');
--      flag_no_hit_last_state<= True;
      
  else 
      if rising_edge(clk) then
      case stm is
    
      when hit =>      
           
          if unsigned(ptr_window_i) /= 0 then
          wr_shifted <= shift_right(unsigned(ptr_window_i), 1 );
          
          else
          wr_shifted <= unsigned(ptr_window_i);
          end if;
          
          if trigger = '1' then
             
             flag1 <= true;
            window2read <= std_logic_vector(ptr_window_i);
             
             
          else
             flag1 <= false;
             -- window2read <= (others=>'X') ;         
          end if;
          
          stm <= hit2;
          
               
          
          

     when hit2 =>      
           if trigger = '1' then
              flag2 <= true;
             window2read <= std_logic_vector(ptr_window_i);

           else
              flag2 <= false;          
                -- window2read <= (others=>'X') ;         
          end if;
           stm <= hit3;
         
      when hit3 =>      
            if trigger = '1' then
               flag3 <= true;
              window2read <= std_logic_vector(ptr_window_i);

            else
               flag3 <= false;          
                -- window2read <= (others=>'X') ;         
           end if;
        stm <= hit4;
            
           
       when hit4 =>      
             if trigger = '1' then
                flag4 <= true;
              window2read <= std_logic_vector(ptr_window_i);

             else
                flag4 <= false;          
                -- window2read <= (others=>'X') ;         
            end if;
           stm <= hit5;
                     
       
        when hit5 =>      
              if trigger = '1' then
                 flag5 <= true;
               window2read <= std_logic_vector(unsigned(ptr_window_i) + 1);
              else
                 flag5 <= false;          
                -- window2read <= (others=>'X') ;         
             end if;
           stm <= hit6;
        
        when hit6 =>      
           if trigger = '1' then
              flag6 <= true;
               window2read <= std_logic_vector(unsigned(ptr_window_i) + 1);
           else
              flag6 <= false;          
                -- window2read <= (others=>'X') ;         
           end if;
             stm <= hit7;

   
        when hit7 => 
     --      if flag
           
           
           
           if trigger = '1' then
              flag7 <= true;
               window2read <= std_logic_vector(unsigned(ptr_window_i) + 1);
          else
              flag7 <= false;          
                -- window2read <= (others=>'X') ;         
           end if;
             stm <= hit8;
             
             
             
       
       when hit8 =>      
             
              if unsigned(ptr_window_i) /=  0 then
            
              wr_shifted <= shift_right(unsigned(ptr_window_i), 1  ) +1;

              else
              wr_shifted <= unsigned(ptr_window_i )  + 1 ;
              end if;
            if trigger = '1' then
               flag8 <= true;
               window2read <= std_logic_vector(unsigned(ptr_window_i) + 1);
           else
               flag8 <= false;
                -- window2read <= (others=>'X') ;         
            end if;
            stm <= hit9;
  
       when hit9 =>      
             if trigger = '1' then
                flag9<= true;
              window2read <= std_logic_vector(unsigned(ptr_window_i) +2); 
            else
                flag9<= false;          
                -- window2read <= (others=>'X') ;         
            end if;
             stm <= hit10;
           
        when hit10 =>      
              if trigger = '1' then
                 flag10 <= true;
             window2read <= std_logic_vector(unsigned(ptr_window_i)+2); 
              else
                 flag10 <= false;          
                -- window2read <= (others=>'X') ;         
             end if;
          stm <= hit11;
          
         when hit11 =>      
               if trigger = '1' then
                  flag11 <= true;
                  window2read <= std_logic_vector(unsigned(ptr_window_i)+2); 
               else
                  flag11<= false;          
                -- window2read <= (others=>'X') ;         

              end if;
             stm <= hit12;
                      
          when hit12 =>      
                if trigger = '1' then
                   flag12 <= true;
                   window2read <= std_logic_vector(unsigned(ptr_window_i)+2);    
                   else
                   flag12 <= false; 
               -- window2read <= (others=>'X') ;         
         
               end if;
             stm <= hit13;
          
          when hit13 =>      
             if trigger = '1' then
                flag13 <= true;
            window2read <= std_logic_vector(unsigned(ptr_window_i)+3); 
             else
                flag13 <= false;
                -- window2read <= (others=>'X') ;         
          
             end if;
               stm <= hit14;
  
          when hit14 =>      
             if trigger = '1' then
                flag14 <= true;
            window2read <= std_logic_vector(unsigned(ptr_window_i)+3); 
             else
                flag14 <= false; 
                -- window2read <= (others=>'X') ;         
             end if;
               stm <= hit15;             

          when hit15 =>      
             ptr_window_trans_i <= ptr_window_i;
             if trigger = '1' then
                flag15 <= true;
            window2read <= std_logic_vector(unsigned(ptr_window_i)+3); 
             else
                flag15 <= false;          
             end if;
               stm <= wr_add;
                 
          when wr_add =>
   
          if trigger = '1' then
          
               window2read <= std_logic_vector(unsigned(ptr_window_i)+3) ; 
               if full_fifo = '0' then
                   ptr_window_i <= std_logic_vector(unsigned(ptr_window_trans_i) + 4);       
                     if wr_shifted /= 255 then  
                          wr_shifted <= unsigned(wr_shifted+1);
                     else
                          wr_shifted <= (others => '0');
                     end if;

--                   if unsigned(ptr_window_trans_i) /= 0 then
--                         ptr_sub_i <= shift_right(unsigned(ptr_window_trans_i), 2  )+1;
--                   else
--                         ptr_sub_i <= unsigned(ptr_window_trans_i) + 1;                
--                   end if;                          
                   stm<= hit;
               else 
                   stm<= hit;
               end if;              
           else  
  --           flag_no_hit_last_state<= True;

             if flag1 or flag2 or flag3 or flag4 or flag5 or flag6  or flag7 or flag8 or flag9 or flag10 or flag11 or flag12 or flag13  or flag14 or flag15 = true then
                 if full_fifo = '0' then

                     ptr_window_i <= std_logic_vector(unsigned(ptr_window_trans_i) + 4);
                     if wr_shifted /= 255 then  
                               wr_shifted <= unsigned(wr_shifted+1);
                          else
                               wr_shifted <= (others => '0');
                          end if;

--                   if unsigned(ptr_window_trans_i) /= 0 then
--                         ptr_sub_i <= shift_right(unsigned(ptr_window_trans_i), 2  )+1;
--                   else
--                         ptr_sub_i <= unsigned(ptr_window_trans_i) + 1;
--                   end if;                  
                     stm<= hit;
                 else   
                     stm<= hit;                 
                 end if;
             else  --- NO HIT 
                 stm<= hit;
                 wr_shifted <= unsigned(wr_shifted-1);
  --               flag_no_hit_last_state<= True;

             end if;             
           stm <= hit;
        end if;       
           end case;
   end if;
      
end if;

end process p_sm;
ptr_window <= ptr_window_i;

----------------------------------
-- Dummy signal for simulations
----------------------------------

p_counter: process(clk, RST)
begin
if RST = '0' then
    counter_i <= (others=> '0');
 
else
    if rising_edge(clk) then
        if counter_i < "111" then
           counter_i <= std_logic_vector(unsigned(counter_i) + 1);
        else 
           counter_i <= "000";

        end if;
    end if;
 end if;

end process p_counter;

counter <= counter_i;

--wr <= wr_shifted;

WR_RS <= std_logic_vector(wr_shifted(1 downto 0));

WR_CS <= std_logic_vector(wr_shifted(7 downto 2));

----------------------------------
-- Dummy signal for simulations
----------------------------------
p_sstin: process(clk,RST)

begin
if RST = '0' then
    sstin_i <= '0';
else
    if rising_edge(clk) then
        case counter_i is 
        when "000" =>
            sstin_i <= '0';
        when "001" =>
            sstin_i <= '0';
        when "010" =>
            sstin_i <= '0';
        when "011" =>
            sstin_i <= '1';
        when "110" => 
           sstin_i <= '1';
        when "111"=> 
           sstin_i <= '0';   
        when others =>
            sstin_i <= '1';
        end case;
     
     end if;
    
 end if;
 

end process p_sstin;

sstin <= sstin_i;

----------------------------------
-- One-cycle signal for the enable_write
----------------------------------

p_enableWrite : process(clk,RST)
begin
if RST = '0' then
      stm_comp <= A;
      saved_i <= (others=> '0');
      enable_write_i <= '0';
  --    rd_add_i <= (OTHERS =>'0');


else

    if rising_edge(clk) then
        if window2read /= saved_i then
            enable_write_i <= '1';
            saved_i <= window2read;
          --  rd_add_i <=  window2read;
        else
            enable_write_i <= '0';
        end if;
     end if;   
end if;

end process p_enableWrite;

RD_add <= window2read;
enable_write<= enable_write_i;




end architecture;