/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  10


const int16_t conv1d_56_bias[CONV_FILTERS] = {-1, 22, 15, 22, 3, 8, 1, 23, 0, 17, -3, 27, 17, -3, -14, 17}
;

const int16_t conv1d_56_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{108, -10, 60, -23, 101, 60, 75, 51, 44, 105}
}
, {{78, -80, -96, -110, -43, -65, -78, -36, -12, 48}
}
, {{-23, -64, -57, -88, -109, 59, 26, -40, -57, -90}
}
, {{-97, -17, -28, 65, -65, 0, 53, 105, 3, 15}
}
, {{108, 32, 48, 33, -8, 23, 80, -84, 17, -41}
}
, {{3, 72, -103, -102, 53, -24, 39, -38, 16, -4}
}
, {{88, -7, 82, 23, -69, -81, -40, -114, -27, -94}
}
, {{-49, -93, -45, 63, -73, -17, -97, -25, 65, 7}
}
, {{-86, 72, 84, 23, 83, 80, -1, 22, 29, -38}
}
, {{96, 93, 49, -43, 44, 25, -80, -80, 88, 26}
}
, {{-79, 75, -47, 83, -3, -77, 9, -34, -36, 74}
}
, {{94, 13, 5, -96, 49, -2, -72, -86, -70, 21}
}
, {{-86, 83, 73, -52, 72, 65, 68, 73, -48, 63}
}
, {{72, -18, -49, 66, 17, -79, 2, 68, -107, -100}
}
, {{61, 63, 95, -29, 43, -92, -85, -73, 35, 27}
}
, {{0, 17, 7, -37, 85, -86, -2, 64, -114, -90}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE