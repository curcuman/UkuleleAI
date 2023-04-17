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


const int16_t conv1d_bias[CONV_FILTERS] = {3, 5, 5, 0, 5, -3, -3, 9, 4, -8, -11, 5, 5, 0, 1, 12}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{67, 62, -46, 90, -27, -63, 54, -52, -25, -69}
}
, {{40, 65, -9, 78, -62, 79, 75, 53, 28, 28}
}
, {{-93, -92, -62, -87, -10, 7, -4, -66, -33, 88}
}
, {{109, -73, 28, 53, 20, 98, 72, -39, 10, 31}
}
, {{-68, -6, -41, 8, 93, 43, -28, -44, 31, 88}
}
, {{-3, -74, 11, -72, 94, 77, -62, 74, 43, -9}
}
, {{27, 8, -77, 79, 29, 8, 40, 66, -47, 96}
}
, {{47, -82, -90, 55, 80, 60, 46, 19, 12, 39}
}
, {{-87, -75, 55, -27, 29, -4, 4, -65, -12, -88}
}
, {{-73, -36, -47, 71, -80, 82, 76, 3, 90, -7}
}
, {{-53, 49, -59, 44, -63, -97, 89, 68, 100, -58}
}
, {{-26, 47, 87, 7, 20, -17, 15, -102, 29, -91}
}
, {{-80, -95, -62, -31, -79, 5, 12, 55, 86, -72}
}
, {{10, 70, 62, -34, -50, 76, 94, -84, 102, -3}
}
, {{16, 45, 65, -78, -1, -94, 70, -93, -37, 43}
}
, {{45, -21, -74, -8, -89, -58, -67, -77, 0, 48}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE