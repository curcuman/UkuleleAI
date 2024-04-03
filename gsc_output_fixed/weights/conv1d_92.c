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


const int16_t conv1d_92_bias[CONV_FILTERS] = {-5, 16, 4, 7, 10, 17, 16, 59, 24, 4, 0, 25, 47, 27, -22, 21}
;

const int16_t conv1d_92_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{53, 11, -94, -70, 79, 48, 78, -92, -12, -61}
}
, {{-54, 75, 23, 114, 43, 12, 110, 88, -2, 13}
}
, {{73, 30, -51, -15, -45, -68, -24, -11, 4, 76}
}
, {{133, 30, 125, 17, 109, -23, 85, 88, -49, -23}
}
, {{-115, -95, 19, 51, -33, -123, -31, -95, -48, 72}
}
, {{28, -15, -43, 65, 61, 112, -14, 105, 51, 7}
}
, {{73, -10, 59, 67, 17, 77, 106, 113, -28, 100}
}
, {{-59, -12, -99, -109, 37, -6, -15, 21, 7, 24}
}
, {{-34, -76, -87, 41, 57, -84, 28, -1, 70, 86}
}
, {{-14, -57, -1, 22, -42, 3, 9, -98, 3, -98}
}
, {{26, 68, 21, 30, -25, -77, -74, 51, -75, 49}
}
, {{0, 28, -87, -88, -95, -32, 76, 95, 18, 32}
}
, {{-109, -53, -29, -59, -8, 108, 88, -16, 35, 124}
}
, {{103, 55, -87, -70, -128, -26, -83, -94, -18, -89}
}
, {{104, 76, -9, 95, -13, 64, -34, 31, -97, -87}
}
, {{-4, -13, 51, 98, 15, 36, 58, 8, 105, 100}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE