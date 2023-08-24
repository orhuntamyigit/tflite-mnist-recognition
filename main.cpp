
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.cpp
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/

#include <math.h>
#include <cstdint>
#include "string.h"
#include "conv_model_array_int8.h"
#include "get_image.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

//TFLite Globals
namespace{

	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* model_input = nullptr;
	TfLiteTensor* model_output = nullptr;

	// Create an area of memory to use for input, output, and other TensorFlow
	// arrays. You'll need to adjust this by compiling, running, and looking
	// for errors.
	constexpr int kTensorArenaSize = 128 * 1024; // allocates 128 kB memory
	__attribute__ ((aligned(16)))uint8_t tensor_arena[kTensorArenaSize];

} //namespace
int main(void)

{

	char buf[50];
	int buf_len = 0;
	TfLiteStatus tflite_status;
	uint32_t num_elements, timestamp, duration;

	// Load model array:
	model = tflite::GetModel(conv_model_tflite);

	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
	  printf("Model provided is schema version %d not equal to supported version %d.\n",
			  model->version(), TFLITE_SCHEMA_VERSION);
	  while(1);
	}

	// Pull in only needed operations (should match NN layers). Template parameter
	// <n> is number of ops to be added. Available ops:
	// tensorflow/lite/micro/kernels/micro_ops.h

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    tflite_status = micro_op_resolver.AddConv2D();
    tflite_status = micro_op_resolver.AddMaxPool2D();
    tflite_status = micro_op_resolver.AddReshape();
    tflite_status = micro_op_resolver.AddFullyConnected();
    tflite_status = micro_op_resolver.AddSoftmax();


	if (tflite_status != kTfLiteOk)
	{
	  printf("Could not add layers/operations");
	  while(1);
	}

	// Build an interpreter to run the model with
	static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver,
														tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	tflite_status = interpreter->AllocateTensors();
	if (tflite_status != kTfLiteOk)
	{
		printf("AllocateTensors() failed\n");
		while(1);
	}
	// Assign model input and output buffers (tensors) to pointers
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);
	// Get number of elements in input tensor

	num_elements = model_input->bytes / sizeof(int8_t);
	printf(buf, "Number of input elements: %lu\n", num_elements);

	while (1)
	{
		// Fill input buffer
		for (uint32_t i = 0; i < num_elements; i++)
		{
			model_input->data.int8[i] = example_image[i]; // input is signed integer
		}
		// Run_inference
		tflite_status = interpreter->Invoke();
		if (tflite_status != kTfLiteOk)
		{
			printf("Invoke failed\n");
		}

		// read the output of the network
		for (uint8_t i = 0; i < 10; i++)
		{
			printf("Output %d: %d\r\n", i, model_output->data.int8[i]);
		}

	}

}
