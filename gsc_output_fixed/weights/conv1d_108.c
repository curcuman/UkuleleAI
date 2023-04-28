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


const int16_t conv1d_108_bias[CONV_FILTERS] = {-6, 1, 0, 4, 10, 28, 10, 28, 20, -10, 38, 46, -11, 1, 17, 10}
;

const int16_t conv1d_108_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-56, 63, 72, 11, -6, 72, -6, -37, 63, -6}
}
, {{-77, -34, -46, -56, -70, -79, -42, -95, 4, -116}
}
, {{117, 71, 40, -36, -31, 43, -48, -68, -27, -89}
}
, {{89, 48, -59, -60, -27, 17, -70, -73, -22, 70}
}
, {{7, -49, 110, 52, 77, -19, 88, -23, 2, 105}
}
, {{-87, -71, -30, -46, 72, -100, -104, 34, 40, -68}
}
, {{78, 83, -30, -48, -33, 23, 16, -8, 79, 98}
}
, {{-88, -77, -95, -81, -75, 6, -37, 63, 94, -31}
}
, {{68, 32, 77, 109, 78, -61, 47, 95, -61, 44}
}
, {{8, 94, 37, -46, 15, 105, 10, -93, 77, -54}
}
, {{-11, 11, -101, -62, -66, -72, 0, -81, -92, -91}
}
, {{-126, -29, 3, -9, -21, 62, -63, 106, 99, 91}
}
, {{88, -48, 59, 0, -104, 33, -66, -54, 59, 56}
}
, {{21, 58, 46, 107, 98, -54, 31, 102, 97, 84}
}
, {{-4, 62, -107, -33, -89, -71, -83, 59, 31, -43}
}
, {{-74, -30, -63, 84, -90, -6, -77, 100, 62, 15}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE