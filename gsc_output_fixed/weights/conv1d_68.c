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


const int16_t conv1d_68_bias[CONV_FILTERS] = {9, 10, 21, 17, 17, 12, 0, 18, 2, 41, 30, 15, 17, 30, 11, 11}
;

const int16_t conv1d_68_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{44, -8, 125, 32, 3, 125, 43, 49, 89, -12}
}
, {{-48, 19, -58, 41, -91, -75, -6, -49, 100, 53}
}
, {{88, 88, 93, 44, 99, -10, 26, 76, -26, 0}
}
, {{-77, -106, 39, 64, 83, -52, 60, -38, -53, 95}
}
, {{-30, -48, -52, -5, -63, -76, -46, -115, -88, 65}
}
, {{-22, 15, 106, 36, 78, -21, -12, 93, 17, 31}
}
, {{28, -104, -12, 42, -85, -17, -30, -27, -16, 49}
}
, {{-79, 63, -50, 73, -57, 50, -10, 11, -39, 99}
}
, {{-15, 9, 109, 22, 8, -70, 53, 86, -61, -97}
}
, {{-123, -118, 52, -46, 85, 26, 52, 38, -15, 45}
}
, {{-98, 30, -41, 27, 79, -28, -61, -4, 47, 48}
}
, {{43, 77, 18, 12, -6, 101, 93, 64, 23, -82}
}
, {{-10, -8, 80, -44, 91, -74, 14, 99, 13, 105}
}
, {{44, 72, 39, 76, 105, 101, 100, -79, 39, -93}
}
, {{101, 110, -71, -61, -7, -38, 14, 91, 0, -35}
}
, {{-29, 42, -59, -73, -75, -47, 64, 95, 93, -26}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE