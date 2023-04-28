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


const int16_t conv1d_48_bias[CONV_FILTERS] = {13, 23, 33, 0, 10, 0, 9, -1, 9, 3, 6, 3, 8, 7, 11, 2}
;

const int16_t conv1d_48_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-31, -68, -71, -9, -30, 36, 49, -104, -5, -122}
}
, {{-65, 119, 118, 125, 18, 61, 73, 60, 73, -16}
}
, {{-113, -51, 88, 63, -67, 6, 53, 113, 3, 121}
}
, {{55, -66, 101, 87, 85, 82, -77, -104, 9, -87}
}
, {{-39, -34, 62, 61, -37, -6, 38, -17, 25, 76}
}
, {{59, 36, 54, 79, 19, 13, -39, 27, -22, 21}
}
, {{-14, -88, -71, -85, -48, -44, 94, 37, -68, 105}
}
, {{48, -68, 94, 15, 44, -53, -116, -21, -17, -11}
}
, {{1, -82, 60, -57, 55, -99, -30, -69, -47, -39}
}
, {{31, 67, 50, 29, -58, 26, -54, 72, 24, -4}
}
, {{66, 13, 9, -72, -81, 91, -24, 11, 42, -51}
}
, {{7, 14, -100, 27, -83, 19, 106, 73, 70, -33}
}
, {{23, 86, 72, 51, -85, 94, -59, -89, -7, 13}
}
, {{-7, -9, -92, 0, 1, -32, -61, -80, -11, 39}
}
, {{-58, -91, -53, -73, -12, 86, 106, 83, -45, 7}
}
, {{-11, 45, 108, -55, -56, 91, -13, -63, -83, -98}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE