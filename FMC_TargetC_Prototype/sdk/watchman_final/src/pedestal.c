/**
 * @file 	pedestal.c
 * @author	Anthony Schluchin
 * @date	18th December 2018
 * @version 0.0
 */

#include "pedestal.h"

/**************** Extern global variables ****************/
/*********************************************************/
/** @brief Array containing registers of AXI-lite */
extern int* regptr;
/** @brief Array containing the pedestal correction for every sample */
extern uint16_t  pedestal[512][16][32];
/** @brief Array containing raw data of the whole array */
extern uint16_t  data_raw[512][16][32];
/** @brief Flag raised when AXI-DMA has an error */
extern volatile bool flag_axidma_error;
/** @brief Flag raised when AXI-DMA has finished an transfer, in OnDemand mode */
extern volatile bool flag_axidma_rx_done;
/** @brief Flag raised when the Triple Timer Counter overflows */
extern volatile bool flag_ttcps_timer;
/** @brief Flag raised when the SCU timer overflows*/
extern volatile bool flag_scu_timer;
/** @brief Instance of the device watchdog */
extern XScuWdt WdtScuInstance;


/****************************************************************************/
/**
* @brief	Calculate the pedestal value for every memory location in the TARGET C
*
* @param	-
*
* @return	XST_SUCCESS or XST_FAILURE (defined in xstatus.h)
*
* @note		-
*
****************************************************************************/
int init_pedestals(void){

	uint64_t sqr_val[4][16][32];
	//double rms[4][16][32];
	uint32_t data[4][16][32];
	int window, window_index;
	int timeout;
	int i,j,count,pair, nmbrwindows;
	int ped_sample;
	int ped_channel;
	int avg = 10;

	/* Create an element for the DMA */
	data_list* tmp_ptr  = (data_list *)malloc(sizeof(data_list));
	if(!tmp_ptr){
		printf("malloc for tmp_ptr failed in function, %s!\r\n", __func__);
		return XST_FAILURE;
	}
	tmp_ptr->next = NULL;
	tmp_ptr->previous = NULL;
    nmbrwindows = 4;
	/* To avoid small offset problem , read window by pair, so the address change every time */
	for(window_index=0; window_index<512; window_index+=nmbrwindows){
		for(pair=0; pair<nmbrwindows; pair++){
			for(i=0; i<16; i++){
				for(j=0; j<32; j++){
					data[pair][i][j] = 0;
					sqr_val[pair][i][j] = 0;
				}
			}
		}
		for(count=0; count< avg; count++){
			//window = window_index;
			for(pair=0; pair<nmbrwindows; pair++){		// do the pedestal on pair of window to avoid the small offset
				window =  window_index + pair;
				/* Give the element's address to the DMA */
				XAxiDma_SimpleTransfer_hm((UINTPTR)tmp_ptr->data.data_array, SIZE_DATA_ARRAY_BYT);

				/* Initiate transfer and measure */
				regptr[TC_FSTWINDOW_REG] = window;
				regptr[TC_NBRWINDOW_REG] = 1;
				regptr[TC_Delay_UpdateWR] = 0;
				ControlRegisterWrite(SMODE_MASK ,ENABLE);
				ControlRegisterWrite(SS_TPG_MASK ,ENABLE);
				ControlRegisterWrite(WINDOW_MASK,ENABLE);
				usleep(50);
				ControlRegisterWrite(WINDOW_MASK,DISABLE); // PL side starts on falling edge

				/* Wait on DMA transfer to be done */
				timeout = 200000; // 10sec
				do{
					/* If needed, update timefile */
					if(flag_ttcps_timer){
						update_timefile();
						flag_ttcps_timer = false;
					}

					/* If needed, reload watchdog's counter */
					if(flag_scu_timer){
						XScuWdt_RestartWdt(&WdtScuInstance);
						flag_scu_timer = false;
					}

					/* The DMA had a problem */
					if(flag_axidma_error){
						printf("Error with DMA interrupt: TPG !\r\n");
						return XST_FAILURE;
					}

					usleep(50);
					timeout--;
				}while(timeout && (!flag_axidma_rx_done));

				/* Update the cache with the data written by the DMA */
				Xil_DCacheInvalidateRange((UINTPTR)tmp_ptr->data.data_array, SIZE_DATA_ARRAY_BYT);

				/* DMA did not respond */
				if(timeout <= 0){
					if(flag_axidma_rx_done) xil_printf("done \r\n");
					printf("\r\nwindow = %d | count = %d\r\n", window, count);
					printf("wdo_time: %d\r\n", (uint)tmp_ptr->data.data_struct.wdo_time);
					printf("PL_spare: %d\r\n", (uint)tmp_ptr->data.data_struct.PL_spare);
					printf("info: 0x%X\r\n", (uint)tmp_ptr->data.data_struct.info);
					printf("wdo_id: %d\r\n", (uint)tmp_ptr->data.data_struct.wdo_id);
					for(j=1; j<32; j++){
						for(i=0; i<16; i++){
							printf("%d\t", (uint)tmp_ptr->data.data_struct.data[i][j]);
						}
						printf("\r\n");
					}
					printf("Timeout on window %d: pedestal initialization failed!\r\n", window);
					return XST_FAILURE;
				}
				else flag_axidma_rx_done = false;

    //           printf("window id= %d\r\n",(uint)tmp_ptr->data.data_struct.wdo_id);

				/* Test the returned values */
                if(tmp_ptr->data.data_struct.wdo_id != window){
					printf("window id is wrong! window = %d | wdo_id = %d | pair = %d | count = %d\r\n", window, (uint)tmp_ptr->data.data_struct.wdo_id, pair, count);
					return XST_FAILURE;
				}
				else{
					/* If data valid, calculate the average */
					for(i=0; i<16; i++){
						for(j=0; j<32; j++) {
							data[pair][i][j] += tmp_ptr->data.data_struct.data[i][j];
							sqr_val[pair][i][j] += tmp_ptr->data.data_struct.data[i][j]*tmp_ptr->data.data_struct.data[i][j];
						}
					}
				}
				/* Release the DMA */
				ControlRegisterWrite(PSBUSY_MASK,DISABLE);
			}
		}

		//window = window_index;
		for(pair=0; pair<nmbrwindows; pair++){
			window = window_index + pair;
			for(i=0; i<16; i++){
				for(j=0; j<32; j++){
					/* Divide the average by avg to have the pedestal value */
					pedestal[window][i][j]= data[pair][i][j]/avg;
					sqr_val[pair][i][j] = sqr_val[pair][i][j]/avg;
				//	rms[pair][i][j] = sqrt(sqr_val[pair][i][j] - (pedestal[window][i][j]*pedestal[window][i][j]));
					//printf("%d, ",pedestal[window][i][j]);
				}
			}
//			if(window == 0){
//				printf("RMS values\r\n");
//				for(j=0; j<32; j++){
//					for(i=0; i<16; i++) printf("%lf\t", rms[pair][i][j]);
//					printf("\r\n");
//				}
//			}
		}
	}

	for (ped_channel = 0; ped_channel <16; ped_channel++ ){
        printf("channel = %d,", ped_channel);
		for (ped_sample=0; ped_sample<32; ped_sample++ ){
			printf("%d,",pedestal[0][ped_channel][ped_sample]);

		}
		printf("\r\n");

	}


	free(tmp_ptr);

	return XST_SUCCESS;
}


/*
**************************************************************************
*
* @brief	Get pedestals using get_windowsRaw
*
* @param	number of average, avg; number of windows for step, nmbrofWindows
*
* @return	Void function, updating data_raw and pedestal global variables
*
* @note		-
*
***************************************************************************
*/


int get_pedestal(int avg, int nmbrofWindows){

int i,j,window,channel,sample;

printf("Arrays Initialization\r\n");
for(window = 0; window< 512; window++ ){
	for(channel = 0; channel< 16; channel++ ){
		for(sample = 0; sample< 32; sample++ ){
			data_raw[window][channel][sample] = 0;
     //       usleep(10);
	//		printf("%d\r\n", data_raw[window][channel][sample]);

	}
}
}



for(window = 0; window< 512; window++ ){
	for(channel = 0; channel< 16; channel++ ){
		for(sample = 0; sample< 32; sample++ ){
			pedestal[window][channel][sample] = 0;
		//	usleep(10);
		//	printf("%.2f\r\n", pedestal[window][channel][sample]);

	}
}
}


printf("Getting data");

for (i=0; i<avg; i++ ){
	for(j=0; j<512; j+=4){

         if (get_windowsRaw(j,nmbrofWindows)== XST_SUCCESS);
             else { printf("get Windows raw failed\r\n");
    //     usleep(300);
         }

	}
  //  printf("avg %d of  %d\r\n", i, avg);

}

	 for(window=0; window<512; window++){
			for(channel=0; channel<16; channel++){
				for(sample = 0; sample <32;sample++){
					pedestal[window][channel][sample] = data_raw[window][channel][sample] /avg ;

	     		}
	     	}
	     }

for (window=0; window< 3; window++){
	 for(sample = 0; sample <32;sample++){
	 	printf("%d\r\n",data_raw[window][2][sample]);
	 }
	 for(sample = 0; sample <32;sample++){
	 	printf("%d\r\n",pedestal[window][2][sample]);
	 }
}


	 return XST_SUCCESS;


}

