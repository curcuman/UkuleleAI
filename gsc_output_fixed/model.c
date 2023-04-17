/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d_40.c" // InputLayer is excluded
#include "conv1d_32.c"
#include "weights/conv1d_32.c" // InputLayer is excluded
#include "max_pooling1d_41.c" // InputLayer is excluded
#include "conv1d_33.c"
#include "weights/conv1d_33.c" // InputLayer is excluded
#include "max_pooling1d_42.c" // InputLayer is excluded
#include "conv1d_34.c"
#include "weights/conv1d_34.c" // InputLayer is excluded
#include "max_pooling1d_43.c" // InputLayer is excluded
#include "conv1d_35.c"
#include "weights/conv1d_35.c" // InputLayer is excluded
#include "max_pooling1d_44.c" // InputLayer is excluded
#include "average_pooling1d_8.c" // InputLayer is excluded
#include "flatten_8.c" // InputLayer is excluded
#include "dense_16.c"
#include "weights/dense_16.c" // InputLayer is excluded
#include "dense_17.c"
#include "weights/dense_17.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_17_output_type dense_17_output) {

  // Output array allocation
  static union {
    max_pooling1d_40_output_type max_pooling1d_40_output;
    max_pooling1d_41_output_type max_pooling1d_41_output;
    max_pooling1d_42_output_type max_pooling1d_42_output;
    max_pooling1d_43_output_type max_pooling1d_43_output;
    max_pooling1d_44_output_type max_pooling1d_44_output;
    dense_16_output_type dense_16_output;
  } activations1;

  static union {
    conv1d_32_output_type conv1d_32_output;
    conv1d_33_output_type conv1d_33_output;
    conv1d_34_output_type conv1d_34_output;
    conv1d_35_output_type conv1d_35_output;
    average_pooling1d_8_output_type average_pooling1d_8_output;
    flatten_8_output_type flatten_8_output;
  } activations2;


  //static union {
//
//    static input_9_output_type input_9_output;
//
//    static max_pooling1d_40_output_type max_pooling1d_40_output;
//
//    static conv1d_32_output_type conv1d_32_output;
//
//    static max_pooling1d_41_output_type max_pooling1d_41_output;
//
//    static conv1d_33_output_type conv1d_33_output;
//
//    static max_pooling1d_42_output_type max_pooling1d_42_output;
//
//    static conv1d_34_output_type conv1d_34_output;
//
//    static max_pooling1d_43_output_type max_pooling1d_43_output;
//
//    static conv1d_35_output_type conv1d_35_output;
//
//    static max_pooling1d_44_output_type max_pooling1d_44_output;
//
//    static average_pooling1d_8_output_type average_pooling1d_8_output;
//
//    static flatten_8_output_type flatten_8_output;
//
//    static dense_16_output_type dense_16_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_40(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_40_output
  );
 // InputLayer is excluded 
  conv1d_32(
    
    activations1.max_pooling1d_40_output,
    conv1d_32_kernel,
    conv1d_32_bias,
    activations2.conv1d_32_output
  );
 // InputLayer is excluded 
  max_pooling1d_41(
    
    activations2.conv1d_32_output,
    activations1.max_pooling1d_41_output
  );
 // InputLayer is excluded 
  conv1d_33(
    
    activations1.max_pooling1d_41_output,
    conv1d_33_kernel,
    conv1d_33_bias,
    activations2.conv1d_33_output
  );
 // InputLayer is excluded 
  max_pooling1d_42(
    
    activations2.conv1d_33_output,
    activations1.max_pooling1d_42_output
  );
 // InputLayer is excluded 
  conv1d_34(
    
    activations1.max_pooling1d_42_output,
    conv1d_34_kernel,
    conv1d_34_bias,
    activations2.conv1d_34_output
  );
 // InputLayer is excluded 
  max_pooling1d_43(
    
    activations2.conv1d_34_output,
    activations1.max_pooling1d_43_output
  );
 // InputLayer is excluded 
  conv1d_35(
    
    activations1.max_pooling1d_43_output,
    conv1d_35_kernel,
    conv1d_35_bias,
    activations2.conv1d_35_output
  );
 // InputLayer is excluded 
  max_pooling1d_44(
    
    activations2.conv1d_35_output,
    activations1.max_pooling1d_44_output
  );
 // InputLayer is excluded 
  average_pooling1d_8(
    
    activations1.max_pooling1d_44_output,
    activations2.average_pooling1d_8_output
  );
 // InputLayer is excluded 
  flatten_8(
    
    activations2.average_pooling1d_8_output,
    activations2.flatten_8_output
  );
 // InputLayer is excluded 
  dense_16(
    
    activations2.flatten_8_output,
    dense_16_kernel,
    dense_16_bias,
    activations1.dense_16_output
  );
 // InputLayer is excluded 
  dense_17(
    
    activations1.dense_16_output,
    dense_17_kernel,
    dense_17_bias, // Last layer uses output passed as model parameter
    dense_17_output
  );

}
