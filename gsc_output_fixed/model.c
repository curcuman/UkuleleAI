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
#include "max_pooling1d_145.c" // InputLayer is excluded
#include "conv1d_116.c"
#include "weights/conv1d_116.c" // InputLayer is excluded
#include "max_pooling1d_146.c" // InputLayer is excluded
#include "conv1d_117.c"
#include "weights/conv1d_117.c" // InputLayer is excluded
#include "max_pooling1d_147.c" // InputLayer is excluded
#include "conv1d_118.c"
#include "weights/conv1d_118.c" // InputLayer is excluded
#include "max_pooling1d_148.c" // InputLayer is excluded
#include "conv1d_119.c"
#include "weights/conv1d_119.c" // InputLayer is excluded
#include "max_pooling1d_149.c" // InputLayer is excluded
#include "average_pooling1d_29.c" // InputLayer is excluded
#include "flatten_29.c" // InputLayer is excluded
#include "dense_58.c"
#include "weights/dense_58.c" // InputLayer is excluded
#include "dense_59.c"
#include "weights/dense_59.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_59_output_type dense_59_output) {

  // Output array allocation
  static union {
    max_pooling1d_145_output_type max_pooling1d_145_output;
    max_pooling1d_146_output_type max_pooling1d_146_output;
    max_pooling1d_147_output_type max_pooling1d_147_output;
    max_pooling1d_148_output_type max_pooling1d_148_output;
    max_pooling1d_149_output_type max_pooling1d_149_output;
    dense_58_output_type dense_58_output;
  } activations1;

  static union {
    conv1d_116_output_type conv1d_116_output;
    conv1d_117_output_type conv1d_117_output;
    conv1d_118_output_type conv1d_118_output;
    conv1d_119_output_type conv1d_119_output;
    average_pooling1d_29_output_type average_pooling1d_29_output;
    flatten_29_output_type flatten_29_output;
  } activations2;


  //static union {
//
//    static input_30_output_type input_30_output;
//
//    static max_pooling1d_145_output_type max_pooling1d_145_output;
//
//    static conv1d_116_output_type conv1d_116_output;
//
//    static max_pooling1d_146_output_type max_pooling1d_146_output;
//
//    static conv1d_117_output_type conv1d_117_output;
//
//    static max_pooling1d_147_output_type max_pooling1d_147_output;
//
//    static conv1d_118_output_type conv1d_118_output;
//
//    static max_pooling1d_148_output_type max_pooling1d_148_output;
//
//    static conv1d_119_output_type conv1d_119_output;
//
//    static max_pooling1d_149_output_type max_pooling1d_149_output;
//
//    static average_pooling1d_29_output_type average_pooling1d_29_output;
//
//    static flatten_29_output_type flatten_29_output;
//
//    static dense_58_output_type dense_58_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_145(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_145_output
  );
 // InputLayer is excluded 
  conv1d_116(
    
    activations1.max_pooling1d_145_output,
    conv1d_116_kernel,
    conv1d_116_bias,
    activations2.conv1d_116_output
  );
 // InputLayer is excluded 
  max_pooling1d_146(
    
    activations2.conv1d_116_output,
    activations1.max_pooling1d_146_output
  );
 // InputLayer is excluded 
  conv1d_117(
    
    activations1.max_pooling1d_146_output,
    conv1d_117_kernel,
    conv1d_117_bias,
    activations2.conv1d_117_output
  );
 // InputLayer is excluded 
  max_pooling1d_147(
    
    activations2.conv1d_117_output,
    activations1.max_pooling1d_147_output
  );
 // InputLayer is excluded 
  conv1d_118(
    
    activations1.max_pooling1d_147_output,
    conv1d_118_kernel,
    conv1d_118_bias,
    activations2.conv1d_118_output
  );
 // InputLayer is excluded 
  max_pooling1d_148(
    
    activations2.conv1d_118_output,
    activations1.max_pooling1d_148_output
  );
 // InputLayer is excluded 
  conv1d_119(
    
    activations1.max_pooling1d_148_output,
    conv1d_119_kernel,
    conv1d_119_bias,
    activations2.conv1d_119_output
  );
 // InputLayer is excluded 
  max_pooling1d_149(
    
    activations2.conv1d_119_output,
    activations1.max_pooling1d_149_output
  );
 // InputLayer is excluded 
  average_pooling1d_29(
    
    activations1.max_pooling1d_149_output,
    activations2.average_pooling1d_29_output
  );
 // InputLayer is excluded 
  flatten_29(
    
    activations2.average_pooling1d_29_output,
    activations2.flatten_29_output
  );
 // InputLayer is excluded 
  dense_58(
    
    activations2.flatten_29_output,
    dense_58_kernel,
    dense_58_bias,
    activations1.dense_58_output
  );
 // InputLayer is excluded 
  dense_59(
    
    activations1.dense_58_output,
    dense_59_kernel,
    dense_59_bias, // Last layer uses output passed as model parameter
    dense_59_output
  );

}
