/*
 * neural_net.h
 *
 *  Created on: Aug 2, 2016
 *      Author: Harald Heckmann
 */

#ifndef SRC_NEURAL_NET_H_
#define SRC_NEURAL_NET_H_

#include <stdint.h>
#include <ap_fixed.h>

// define the operation - DO NOT CHANGE - IT WILL REQUIRE A LOT OF ADAPTION
#define OPERATION_INIT 0x0
#define OPERATION_EXECUTE 0x1

// define the maximum amount of nodes
// in each of the three layers
#define MAX_INPUT_NODES 450
#define MAX_HIDDEN_NODES 450
#define MAX_OUTPUT_NODES 450

// define fixed-point types
#define FP_PRECISION 16
#define FP_INTEGER_WIDTH 2

typedef ap_fixed<FP_PRECISION, FP_INTEGER_WIDTH, AP_RND_CONV, AP_SAT> fixed;
//typedef ap_ufixed<FP_PRECISION,FP_INTEGER_WIDTH,AP_RND_CONV,AP_SAT> fixed_uarith

// attention: If you pass more weights, input values or output values,
// it will result in some inputs/outputs not getting transfered.

// transfer data as a stream
#pragma SDS data access_pattern(weights0to1:SEQUENTIAL, weights1to2:SEQUENTIAL,\
		inputvalues:SEQUENTIAL, outputvalues:SEQUENTIAL)

// define, how many data has to be streamed
#pragma SDS data copy(inputvalues[0:inputnodes*op_exec])
#pragma SDS data copy(outputvalues[0:outputnodes*op_exec])
#pragma SDS data copy(weights0to1[0:inputnodes*hiddennodes*op_init])
#pragma SDS data copy(weights1to2[0:hiddennodes*outputnodes*op_init])
// define the memory attributes
#pragma SDS data mem_attribute(inputvalues:NON_PHYSICAL_CONTIGUOUS|CACHEABLE)
#pragma SDS data mem_attribute(outputvalues:NON_PHYSICAL_CONTIGUOUS|CACHEABLE)
#pragma SDS data mem_attribute(weights0to1:NON_PHYSICAL_CONTIGUOUS|CACHEABLE)
#pragma SDS data mem_attribute(weights1to2:NON_PHYSICAL_CONTIGUOUS|CACHEABLE)
// define data mover
#pragma SDS data data_mover(inputvalues:AXIDMA_SG)
#pragma SDS data data_mover(outputvalues:AXIDMA_SG)
#pragma SDS data data_mover(weights0to1:AXIDMA_SG)
#pragma SDS data data_mover(weights1to2:AXIDMA_SG)
// define system port
#pragma SDS data sys_port(inputvalues:AFI)
#pragma SDS data sys_port(outputvalues:AFI)
#pragma SDS data sys_port(weights0to1:AFI)
#pragma SDS data sys_port(weights1to2:AFI)
void neural_net(uint8_t op_init, uint8_t op_exec, uint16_t inputnodes,
		uint16_t hiddennodes, uint16_t outputnodes,
		fixed weights0to1[MAX_INPUT_NODES * MAX_HIDDEN_NODES],
		fixed weights1to2[MAX_HIDDEN_NODES * MAX_OUTPUT_NODES],
		fixed inputvalues[MAX_INPUT_NODES],
		fixed outputvalues[MAX_OUTPUT_NODES]);

// definition of parameters:
// weights0to1 contains all weights from layer0 (input) to layer1 (hidden)
// weights1to2 contains all weights from layer1 (hidden) to layer2 (output)
// input_values contains all input values for the input ondes on layer0
// output_values will contain all results the neural net has calculated
// inputnodes contains a number which describes the amount of:
// a) input values
// b) weights from l0 to l1 in combination with parameter hiddennodes
// hiddennodes contains a number which describes the amount of:
// a) weights from l0 to l1 in combination with parameter inputnodes
// b) weights from l1 to l2 in combination with parameter outputnodes
// outputnodes contains a number which describes the amount of:
// a) output values
// b) weights from l1 to l2 in combination with parameter hiddennodes
// operation descripes the desired operation, INIT or EXECUTE (currently), where:
// a) INIT initializes the BRAM with weights and partitions it to parallelize and pipeline it optimally
// b) EXECUTE transfers input values and calculates the output values with the initialized values from INIT

#endif /* SRC_NEURAL_NET_H_ */
