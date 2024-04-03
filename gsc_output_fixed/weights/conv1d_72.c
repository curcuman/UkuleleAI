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


const int16_t conv1d_72_bias[CONV_FILTERS] = {8, 15, 23, 4, 10, -15, -4, -20, 11, 33, 5, 16, -4, 3, 2, 1}
;

const int16_t conv1d_72_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{26, -66, -51, -69, -84, -109, -62, -77, -73, -112}
}
, {{81, 99, -7, -35, -35, 0, 102, 101, 83, 110}
}
, {{61, 57, 92, -12, -26, 16, 69, 78, 31, 72}
}
, {{27, 103, 32, 23, 98, -28, -9, -85, -101, -107}
}
, {{-101, 27, -57, 53, -80, 82, -88, -76, 52, 25}
}
, {{102, 32, -10, 77, 58, -19, -91, 54, -78, -48}
}
, {{-98, -24, 75, -5, -90, 33, 39, -71, -87, 15}
}
, {{42, 6, -68, -79, 40, -85, -32, -26, 18, -84}
}
, {{34, 28, 90, 53, 112, 57, -49, -28, -18, -92}
}
, {{64, -73, 43, -92, -62, -72, -67, 43, 93, -39}
}
, {{-90, -80, -20, 77, -82, -79, -71, 62, -44, -79}
}
, {{103, 65, -56, 5, 56, -66, -105, 36, -90, -114}
}
, {{-33, -80, -79, 50, -53, -35, -72, 103, 52, 18}
}
, {{31, 31, -62, 21, -94, 55, -12, -93, 19, -13}
}
, {{104, -45, 101, -57, 82, -22, 71, -48, 81, 103}
}
, {{-89, -91, 43, -76, -1, -66, -31, 68, 9, 52}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE