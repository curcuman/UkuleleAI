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


const int16_t conv1d_24_bias[CONV_FILTERS] = {5, 0, 1, -3, 2, 3, 11, -4, -3, 5, -2, 3, 1, 4, 4, 7}
;

const int16_t conv1d_24_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{60, -30, 88, 11, -40, 15, -49, -70, -69, -12}
}
, {{-80, 50, -58, -35, -22, -61, -62, 46, 8, -65}
}
, {{89, 35, 29, -15, -11, 56, 15, 16, 16, -99}
}
, {{-1, 53, -37, -10, 76, -1, 63, 86, -15, 6}
}
, {{-14, -2, 52, 66, -79, -55, 68, 34, -6, 72}
}
, {{34, 96, 20, 87, 44, 72, 7, -73, -48, 47}
}
, {{100, 37, 83, -52, -55, -90, 60, -90, -19, -92}
}
, {{82, -75, 87, 30, 59, 29, -48, 78, 64, 64}
}
, {{-51, 58, -94, -61, -16, 47, -33, -79, -54, -41}
}
, {{-41, 49, 69, -90, -56, -68, 19, 55, 28, 20}
}
, {{84, 34, -96, -83, -53, -3, 82, 73, -9, 68}
}
, {{-72, -69, 49, 92, 37, 50, -89, 99, 26, 11}
}
, {{-51, -53, -51, 92, 75, -34, -31, 27, 0, 2}
}
, {{86, 61, -93, 23, -63, -6, -82, -38, 59, -62}
}
, {{-89, 31, -4, -84, -92, 48, 74, 79, -8, 25}
}
, {{10, -60, -12, -88, -75, -93, 72, 51, 58, 18}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE