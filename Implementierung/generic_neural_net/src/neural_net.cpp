/*
 * neural_net.cpp
 *
 *  Created on: Aug 2, 2016
 *      Author: Harald Heckmann
 */

#include "neural_net.h"




void init_neural_net(fixed bram_weights0to1[MAX_HIDDEN_NODES][MAX_INPUT_NODES],\
		fixed bram_weights1to2[MAX_OUTPUT_NODES][MAX_HIDDEN_NODES],\
		fixed weights0to1[MAX_INPUT_NODES*MAX_HIDDEN_NODES],\
		fixed weights1to2[MAX_HIDDEN_NODES*MAX_OUTPUT_NODES],\
		uint16_t inputnodes, uint16_t hiddennodes, uint16_t outputnodes)
{
#pragma HLS INLINE self
#pragma HLS dataflow
#pragma HLS array_partition variable=bram_weights0to1 cyclic factor=450 dim=1
#pragma HLS array_partition variable=bram_weights1to2 cyclic factor=450 dim=1
//#pragma HLS array_reshape variable=bram_weights0to1 cyclic factor=450 dim=1
//#pragma HLS array_reshape variable=bram_weights1to2 cyclic factor=450 dim=1
	// loop counter
	uint16_t i,j,k,l;

	for (j=0; j<MAX_HIDDEN_NODES; j++) {
		for (i=0; i<MAX_INPUT_NODES; i++) {
#pragma HLS PIPELINE II=1
      // if there is still data to transfer, get it
      if (i < inputnodes && j < hiddennodes) {
        bram_weights0to1[j][i] = weights0to1[j*inputnodes+i];
      } 
      // otherwise initialize the weightmatrix with zeros, so the results of
      // the caluclation will still be correct
      else {
				bram_weights0to1[j][i] = fixed(0.0);
			}
		}
	}

	for (l=0; l<MAX_OUTPUT_NODES; l++) {
		for (k=0; k<MAX_HIDDEN_NODES; k++) {
#pragma HLS PIPELINE II=1
			if (k < hiddennodes && l < outputnodes) {
				bram_weights1to2[l][k] = weights1to2[l*hiddennodes+k];
			} else {
				bram_weights1to2[l][k] = fixed(0.0);
			}
		}
	}
}

void exec_neural_net(fixed bram_weights0to1[MAX_HIDDEN_NODES][MAX_INPUT_NODES],\
		fixed bram_weights1to2[MAX_OUTPUT_NODES][MAX_HIDDEN_NODES],\
		fixed inputvalues[MAX_INPUT_NODES], fixed outputvalues[MAX_OUTPUT_NODES],\
		uint16_t inputnodes, uint16_t hiddennodes, uint16_t outputnodes)
{
#pragma HLS INLINE self
	static fixed register_hiddenvalues[MAX_HIDDEN_NODES];
	static fixed register_inputvalues[MAX_INPUT_NODES];
	static fixed register_outputvalues[MAX_OUTPUT_NODES];
#pragma HLS array_partition variable=register_inputvalues complete
#pragma HLS array_partition variable=register_hiddenvalues complete
#pragma HLS array_partition variable=register_outputvalues complete
	//#pragma HLS array_reshape variable=register_inputvalues complete
	//#pragma HLS array_reshape variable=register_hiddenvalues complete
	//#pragma HLS array_reshape variable=register_outputvalues complete

	uint16_t i,j,k,l;

  // Stream the inputvalues sequentially from the inputstream into registers
  // and automaticly apply rectifier, so the values for the caclulation of the
  // netinput in the next layer are ready
	for (i=0; i<inputnodes; i++) {
	#pragma HLS PIPELINE II=2
		register_inputvalues[i] = inputvalues[i];

    // The following if-branch is the rectifier function
		if (register_inputvalues[i] < fixed(0.0)) {
			register_inputvalues[i] = fixed(0.0);
		}
	}

  // Initialize the registers for the values of the hidden layer fully parallel
	for (k=0; k<MAX_HIDDEN_NODES; k++) {
	#pragma HLS unroll
		register_hiddenvalues[k] = fixed(0.0);
	}

  // Initialize the registers for the values of the output layer fully parallel
	for (l=0; l<MAX_OUTPUT_NODES; l++) {
#pragma HLS unroll
		register_outputvalues[l] = fixed(0.0);
	}

  // Calculation of the netinput for the hidden layer
	for (i=0; i<MAX_INPUT_NODES; i++) {
		if (i > inputnodes) break;
		for (j=0; j<MAX_HIDDEN_NODES; j++) {
#pragma HLS unroll factor=450
			fixed product = register_inputvalues[i] * bram_weights0to1[j][i];
			register_hiddenvalues[j] += product;
		}
	}

	// Apply Rectifier fully parallel
	for (i=0; i<MAX_HIDDEN_NODES; i++) {
#pragma HLS unroll
		if (register_hiddenvalues[i] < fixed(0.0)) {
			register_hiddenvalues[i] = fixed(0.0);
		}
	}

  // At this point, we have got the outputvalues for the hiddenlayer
  // Calculation of the netinput for the output layer
	for (i=0; i<MAX_HIDDEN_NODES; i++) {
		if (i > hiddennodes) break;
		for (j=0; j<MAX_OUTPUT_NODES; j++) {
#pragma HLS unroll factor=450
			fixed product2 = register_hiddenvalues[i] * bram_weights1to2[j][i];
			register_outputvalues[j] += product2;
		}
	}

	// Apply rectifier sequential this time and transfer each output value
  // immediately to the PS when it is calculated, so the calculation and the
  // transfer of the caluclated data happen in parallel
	for (i=0; i<outputnodes; i++) {
#pragma HLS PIPELINE II=1
		outputvalues[i] = ((register_outputvalues[i] < fixed(0.0)) ? fixed(0.0) : register_outputvalues[i]);
	}
}


void neural_net(
		uint8_t op_init,\
		uint8_t op_exec,\
		uint16_t inputnodes,\
		uint16_t hiddennodes,\
		uint16_t outputnodes,\
		fixed weights0to1[MAX_INPUT_NODES*MAX_HIDDEN_NODES],\
		fixed weights1to2[MAX_HIDDEN_NODES*MAX_OUTPUT_NODES],\
		fixed inputvalues[MAX_INPUT_NODES],\
		fixed outputvalues[MAX_OUTPUT_NODES]
)
{
	#pragma HLS INLINE self
	// Define the variables for blockram
	static fixed bram_weights0to1[MAX_HIDDEN_NODES][MAX_INPUT_NODES];
	static fixed bram_weights1to2[MAX_OUTPUT_NODES][MAX_HIDDEN_NODES];

  // if the operation is init, call the BRAM initialisation function block
	if (op_init == 1) {
		init_neural_net(bram_weights0to1, bram_weights1to2, weights0to1,\
						weights1to2, inputnodes, hiddennodes, outputnodes);
	}

  // if the operation is exec, call the KNN calculation function block
	if (op_exec == 1) {
		exec_neural_net(bram_weights0to1, bram_weights1to2, inputvalues,\
				outputvalues, inputnodes, hiddennodes, outputnodes);
	}
}
