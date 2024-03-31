#pragma once

#include "DSPLib.h"
#include "../cnn/cnn_relu.h"
#include "../cnn/cnn_types.h"
#include "../cnn/cnn_utils.h"
#include "../utils/myuart.h"

CNNLayer_t ReLU1[1] = {
{
    .lix = 0,
    .fun = CNN_ReLU, // also verified with CNN_Intermittent_ReLU
    .weights = (Mat_t){
        .data = 0,
        .n = 0,
        .ch = 0,
        .h = 0,
        .w = 0
    },
    .bias = (Mat_t){
        .data = 0,
        .n = 0,
        .ch = 0,
        .h = 0,
        .w = 0
    },
    .ifm = (Mat_t){
        .data = 100,
        .n = 1,
        .ch = 4,
        .h = 32,
        .w = 32
    },
    .ofm = (Mat_t){
        .data = 16484,
        .n = 1,
        .ch = 4,
        .h = 32,
        .w = 32
    },
    .parE = (ExeParams_t){
        .Tn = 2,
        .Tm = 0,
        .Tr = 16,
        .Tc = 32,
        .str = 1,
        .pad = 0,
        .lpOdr = OFM_ORIENTED
    },
    .parP = (PreParams_t){
        .preSz = 1,
    },
    .idxBuf = 0
},
};

CNNModel_t network={
    .Layers       = ReLU1,
    .numLayers = 1,
    .name = "Verify_ReLU"
};

const _q15 INPUTS_DATA[] = {
      -122,  -1147,  -8562,  15295,   8789, -15474,  -4731,  12435, -13484,
     -9446, -10604, -14336, -12057,   6644,   7044,  -6422,  -6310,  -5557,
      1556,   8701,   4393,  11308,  -5605,  -9353,   -324,  12952,  -1663,
      3217,  12990,   2383,   -324,  -7023,  -1453,  -1295,  14048,   3254,
      4335,  -5200,  14625,   7994,  -4951,   -828, -12613,   5149,  -3220,
      2982,   6004,  11901, -15652, -12514,   6326,  -1971, -10850,  -3915,
      6264,   8894,  -6753, -13628,  -2233,   5296,    606,  10055,   4097,
     -5163,   6477, -10433,   4122, -11571,   9830,  14967,  15946,  14989,
    -11107,  -4223, -15702,  -4884,  -7134,  -9385,  -6217,  12340,   5950,
      7871, -11023,  10327,  13605,   2442,  -7947,   -659,  -3371,  11345,
      3216, -13533,  12260,   6839,   9992,   1318,  -2640, -15783, -12389,
     -7920,   1733,  10362,  10656,   4263,  14835,  -3085,  13685,  -6512,
    -15198,  -7241,  -8075,  12628, -10314,  10404,   2859, -14782,  -4147,
     11949,  -8411,  -4410,  -6386, -14400,   3509,   2583,  14155,  -1481,
     -7595,   1562, -10619,  13454,  -6382,  13460,  -7542,   6345, -10967,
     -5176, -11446,  13803, -13662, -13078, -15344,  -5614,  13074,  10847,
     -9564,  -9038,  -2069,   4578,  14083,  14088,   -454,  10792,   7310,
      6829,  -8153,  -9696,   7940,  15728,  11419,  -9623,    861,  -6843,
     -9377,  -6618,  -8399, -10519,  -6626,   1164,   2771,  -1920, -12789,
      5551, -15297, -15426, -13776,  -9415, -11838,   6422,  14545,  -1542,
     -8446,  12084, -12651,  -2831,  10337,   3933,  -2261,  14949,   9606,
     -1618,  -4742,  -5908,  -7266,   8123,   6001,  15877,   -591, -10399,
      4382, -10048,  10478,  16027, -12559,  -6751,  16287, -16291,  14714,
    -12323,   6502, -15694, -16320,  12070,   2213,  -3871,   1620, -11266,
     10985,  13383,  -1368,   6781,  -9646,   1640,  -3796,  -6020,   3053,
      6292,  11165,  16326, -12702, -12009,  13833,   2311, -11355,   5975,
      2134, -14228,  -8463,  -1830,  -5786,   2819,   7413,   6567,  12180,
      1991,   6588,  11571,  10293, -15846,  -9705,   7121,   2942,   3288,
      4949,  -1393,  -4481,  15631,   8994,  -1009,    690,   4714,  -2067,
    -10275,  14602,    -87,    625,  -5926,   1776,  -3396,   3796,  10646,
     -3228,   4353,  10164,  -6569, -13871,  -7299,  15731,  10175,  -8779,
     16134, -12625,  -6496,  11833,  14163,  -6004,  -3815,   -576,   7654,
      6439,    348,  -1091,  -6102,  13574, -15035,   6379, -15383,  14257,
      -163, -10779,  -9431,  14456,  -1651,  -1106,  15930,   3260,  -1471,
     -4608, -15038, -14247,  -3275,   8572, -15643,   1507,  12916,  13333,
     -1180, -10249,  12091,  10287,   5212, -15269, -11104,  12215, -16306,
     14557,   7610, -14642, -15211,  12457, -12851,  -5816,     89, -16343,
    -13948,   1940,  -5495,   3066,   5008, -13915,  11414,  -2760,    240,
      4444,   4704,  -2696,  -7643, -10739, -14221,  -7499, -15802,   7020,
    -14767,   6300,  -6342,  13260,  11251,  -9704,   5472, -14015,  -2905,
      6006, -15165,  13728,  -7167,   8285, -15916,   7644,   5160,  11728,
      2078, -10812,  -3220,   6126, -11803,  15682,   7835, -16215, -14353,
    -14189,   4154, -10628,  -6312, -11030,  -7093,   8180,  -2637,  -8150,
    -13131,   3429, -14888,   -829,   8869, -12780, -10940,   2942,  -9843,
     -9434,   1074,  -4808,  -1812,  15413, -15449, -15529,   2240,  11039,
     -2302,  -1891,  -9534,  -7143, -10579,  12430,   8924,  -4123, -11358,
     12614, -11524, -15607,   6439,   8201,   4024,   -294, -12610,   2378,
      1752, -12338, -12416,   4071,   6011, -12637,   9343, -12630, -16079,
      -902,   3063,   6634,    946,   2459,  14921, -16305,   7987,  -6709,
     -4379,  -6766, -10489,   9721,  -7913,  -4305,   4054,  -9970,  -1516,
     12031,     64,  14866,     82,   -477,  13899,  11227,  12495,  -7463,
     -3469, -13816, -16319,   2862,    111,  -4077,  -1166, -13594,  -7126,
       739,   7247,  -4337,  -4037,   2390,  -9022,   4537,   4690,   3885,
     -8325,  -1085, -15801,   6429,   7917,  -1720,  12429,    981, -15698,
     10013,  11781,  -7994,   3882, -12445,   6938,   7752, -12684,  -8272,
     12265, -15716,   8143,  15859,  -5100,  -9710, -14008,   3260,   3731,
     -4101,  10072,  -2652,    833,  -7980, -15509,   8486, -11329,  -5731,
    -13771,   9856,  -6298, -13428,    241,  -2737,  -5816,  -3485,  12910,
     -6824,  -4051,   3502, -14031,  -2678, -14547, -10673,  -7308,   5949,
    -11522,   -840,  14389,  -2091,  -1555,  11728,  -2747,   9535,  -2492,
     -1684,   6787, -13360, -14872,    455, -15761,  -8038,   3500,  -1413,
     -5152,  12490,   8920,   3315,  -8320,   2149,  13644,  10417,  13812,
      3299,  13788,  15519,   6800,   5388, -11448,  10404, -11955,  -3871,
     -5250,  15555,   2641,  -6552,  -7772,  -1184,   -188,   4700,   3004,
    -14718,  -6731,  -4342,   4390,  -7767,  -9356,  -6768,   3645,  11155,
     -5436,   2164,   9574,   -106,  12914,   3962,   8643,  -8143, -11744,
     -1479,  -8794, -12555, -11671, -11080,  11247, -15333,  -5054,  -2477,
     11052, -13828,    285,  -9285,   2168,  -3323,   4459,   8623, -14254,
      8985,    491, -10856, -10048,   8857,  -3944,  13063,   4565, -15801,
     -9934,   6525,   5664,  10220,  -2939, -14029,  15277, -12820,  -6595,
     12158,   8338,  -3463,  13237, -16202,   5938,  -6643, -15948,   -952,
      -558,  -3155, -10450, -10229,  11171,  -3216, -13329, -14094,  -5963,
    -14702,  -2994,   1623,   6671, -14146,  12767,  -7943,  -7803,  -2563,
     14361,  11842,   -152,    211, -14263, -12879,   9682,  -7442,   -709,
     13599,   7517,   6171,   9616,   9884,  -3676, -14746,   7222,  14350,
      -207,  -1105,  13420,  -4797,  -9095,  14408,   -454, -10042,   7589,
     -6682, -11008, -10884,  11024,  14794,  10838,  13624,   5639,   5933,
     -5618,  -8149, -15854, -14785,   8483,  -2017, -11728,  10366,  -4037,
     -2525,  10968,  -1890,   6964,   2818,  -2345,  -7313,  14736, -13444,
     10061,  13101,  15234,  13211,  12076, -13239, -14910, -12123,  -4799,
      1758, -10837,  16114,  -9612,  -3430,  -6774,  -6410, -15413,  11700,
    -10448,  -7314,   6267,   4573,  15720,  -2252,   8403,   7872,  -1447,
      4848,  -2840,   5786, -10868,  -3035, -15906,  -3939,  -7290, -16352,
    -10553,  -3445,  -9049,  11524,   -882, -13501,   6684,  16228,   7477,
      8877, -14290,   8266,  -5489,  13008, -15276,  16032, -10576,  11210,
    -16150,   4465, -11406, -11556,   7053,  -1283,  14659,    730,   8101,
    -13049,   5933, -11549, -10175, -14319, -13302,  -9019,   4062, -11280,
    -10118,  -9547,  12094,  -4156, -13908,   5599,   2380,   9540,   2370,
     -9763,  -3466,  -6759,  -6652,   -357,   1210,   -636,    -55,    689,
     -1756,   -231,  -2591,  10561,   8249, -10709, -14341, -12384, -11124,
      -443, -14641, -11247,   8308,  15819, -12707,  -9513,   9234,  13094,
     -6861,  11467,  -9926,  12478,   2548,  -5889,  13378,  -1615,  -7705,
     13819,  16092,    290,   -584,   5924,  -6459,   8956,  -1962,   2074,
     -4589,  -5449,  14983,   -121,   8165,  -6639,  -7468,  -3238, -12835,
     14017,  15393,   2055,   6479,    935, -12577,  -3741,   3822, -11062,
    -13669,   -115,  10327,  13805,   5327,   2090, -14231,   2719,  -9535,
    -12815, -13611,   8766,   8403,  -8587,   7358, -14229,  -9064,  13229,
      5901,   3040,  12901, -13296, -15249,  10021,  15864,  -1176,  -9136,
     -2363,   1048,  16207, -14991,  -9312,   3501,   5918, -11737,  13108,
     15275,    463,  -7392,   7620,   9712, -14198,  13558,   4666, -14303,
      8116,  -3261,   3874,   -524, -11670,   9410,    361, -14789,  -4650,
     15559,   9395,  12515,  -5497,   7466,  -3581,  -3000,  -2426, -14667,
     -9644,   -114,    179,   8244,   6256, -10066,  13513,   6249,   4371,
     -8908,   2045,  12698,   6238,   2126,  14675, -16161,  -1819,  13517,
     10022,  10076,  -2174,  -8235, -10358,  13410,  -1302,   1548,   7348,
    -15243,   4320,  14830, -11581,   1714, -12572,  14683,  -6943,  13986,
    -15270,  -3203,   4818, -11916,   6130,  15356,   5409,   5375,  -8970,
      5966,  12291,   7239,  -1378,  14921,  -5274,   4294,   4542,    486,
        26,  16290,   2298, -13885,   8434,  13414,  10562,  -4459, -15844,
     14056,   2147,   4914,  11845,  -2042,   4056,   8737, -13548,  13741,
     -1466, -16321,    225, -15268,   2418,  12744,  -2785,  -8841,   6007,
    -14697,  -8629,  -3504,  11177,   9599,   2165,  -4866, -15524,  16165,
     13548,  15133,  -6824,   4037,  -4789,   7727, -14701,  -3092,  -9727,
     10548,  -6532,   9021,  -6059,  11144,   9987,  -4862, -16238, -12238,
    -15523, -12136,   7395,  -1897, -14680, -11627,  -7868,   2392,   9486,
      7703, -10933,   4261,  -9675,  12512,  -9438,   8425,  11974,    663,
      9420,  -6519,   5105, -15885,   8676,  -4940,   4112, -16099,  12575,
     13709,  13015,  -3699,   5942,  -4397, -15176,   8844,  -5471, -11381,
     11997, -10918,  -4578,  11109,   5313,   4416,   4840,  11094, -10979,
     -8955,  13468,  -5137,   3986,  13330,   4454,  -3275,  -4566,  -4293,
     -7752, -12089,  -6576,  -3414,  -7701, -13682,  -1774, -14370, -15490,
      6161,  -7724, -14675,   3539,  -6999,  14326,   1107,  -9194,  13043,
     14101,   2649, -14607, -13118,  -1747,   8851,  14366,  -2264, -14528,
    -14326, -10639,    353,  -1002,   2870,  -1863,   8420,   1082,  -2366,
      4693,  13260,  10951,   9610,    521,  11095,    527,  -7449, -11024,
      7212,   4768,   4881, -13243,  -7520,   4347,   6602,  13059, -11700,
     -9296,   8230,   2667, -14657, -10162,  -5619,  13592,    812, -14643,
      5272,  -5492,  15890,   -128,  -3210,   4825,  -3112,   5074,   4411,
     -3746,   -689,  -1164, -10074,   -728, -11906,  15903, -13018,  -9978,
      5030,  -3125, -13401,   5541,  10387, -10691, -10299,   5180, -16153,
      2730,  -6934,   -337, -11536,  -4613,   2438,  -3684,  12135,  -2246,
     -3583, -10099,   3879,  15875,  -9327,  11330,  -1021,   -470,    137,
    -12196,   2928,  14312,  -5174,   6711,  -2491,   2188,  10099,  -5509,
    -11451,  -6095, -13493,  -7904, -15309,  12012,  -6047,   2943,  10775,
     -9332, -10528,  -8510,  -1948,    542,   2021,   3774,  -6084,   3682,
       893,   3217,  -4228,   7353,   -942, -12164,  -8936, -15116,  16256,
      2727,  -4852,    803,  -6250,   6978, -14064,  -4242,  -2354,   6485,
     12260,  10622,   4232,  -2062,   6690,   5512,  13674, -13431,  -8262,
      6566,  12418,  -2525,  -5592,   9851,   1871,   5690,   7350,  -3668,
     -6198,  -5978,  -6067,  -1049,   5382,   6219,   4320,  13144,  10308,
     10911,   4511, -13205,   -349,  -8554, -10313, -11324,   8995,    161,
      4104,   1198,  13409,   6774,   1002,  -6490,  -8778,   1284,  14309,
       938,  -3622,   1368,  -5928,  14472, -14750,   2045,  -4211,  -1899,
      3761, -12880,  14116, -13969,  11676,   1287, -13379,  -7447,  -3181,
     11345,   1183,  15046,   2614,  14763,  15786,  -3599,   5235,   9629,
     -7346,  -7184,   -349,   2195,  15707,   7374,  12737,   7651,  14538,
      2028,   2387,  -7970,  15242, -14445, -16104, -13577,  -1563,  13644,
    -13529, -14088,  14821,   9226, -16232,  16344,  -7138,  14240,    675,
     10400,  -5608,   7169,   8856, -11325,   6196,  -5523,  -2161,   6410,
      3461,  -5348,  14731,  12372,   9032,   7940,  -2947,  16377,  15570,
     -3176,   3349,  14326, -10256,   3212,  14774,  12693,   8695, -15875,
      8291,  -3756,   5481,  -8098,   -522,  -5749,   8643,   6085,  15781,
     13452,   7768,   -807,  -9788,   9181,  -6128,  -7876,   6102,  -9859,
    -12949, -14291,  10347,  14729,  13113,  10413, -10144,   7916,  -9342,
     -3294, -15390,   8931,  10474, -16097,   5240, -10269, -15608,  15807,
     -6215,   4700,  -2834,   7972,  -8268,  -5743,    173,  -9799,   3258,
     12801,   2834, -11957, -11885,  -2948,   1324,  15832,  14735,   6378,
      5162,     68,  -1470,   2910,  -1824,  -9853,  -5247,   6970,   4012,
      9428,  13696,  -5567,  16213,  -3412,   4561,   7988, -10601,   1122,
      3121, -11443,   -894,  -8534,  -7833,   3700,  -6254, -13877,  15615,
    -11085,  14375, -10880,  -4990, -16163,  14025,  -1799,  13627, -13157,
      -446, -11861,  -1204,  12934,  -5679,  16148,   8218,   8864, -10123,
       177,  11519,  15371,  11723,  -6272, -12197,  13125,  -1128,   6475,
    -10919, -14631,  15430, -14525, -16201, -11180,   9343,  -8380,  11817,
     -2647,  14340,   6153,  14090, -10640,  12958, -13920,  -5987,  11376,
    -10647,   8165,  -8024, -12386,   4795,   1377,   2742,  -7993,  -6793,
    -13661,   8820, -15828,   8220, -16260,   4044,  -9301,  -1473,   5178,
     -3414,  13475,   1418,   7744, -10421,  13414, -11331,  -3275,  14012,
     11728, -14642, -14487, -10610,  12650,   6169,  -9442, -15831,  14568,
      3807,   1741,  -6541,  -4195,   1423,    596,   3267,   7208,  11127,
      4193,   6089,  14597,  -6841,   2897,   6335,   5420,   5816,   -858,
     -9491,  16378,  -9023,   8646,  -6288,   8497,   3394,  15892,  -7285,
     10185,  -3740,  13644,  -7934,  -5734,  11474,  14887, -13799,   7861,
     14327,  -5529,   3353,   1882,  14229,  -5592,  -4764,  -3912,    737,
     -1812, -15769,  -9235,   9661,   4842,  -9566,  -9193, -16369,   6631,
     -9839, -12607,  -9610,  -1460,  -7067,  10999,   2928, -10582, -10123,
     11647,     91, -12361,  -2841,  -1864, -12862,  15773,   5871,  -9481,
     -7251,  -4404, -13259,  12663,  -7652,  -2214,  -1403,  10477, -16232,
    -10041,  -4715,   1217,   1562,  -8122,  -8376,  -7735,  11168,   3598,
     -3638,  15057,   6649,   6900,  -6944,   6700,  -7784,  -6476,    526,
    -12437, -15044,  -6030, -11826,  15680, -13218,  12978,   3107,  12441,
       303,  -8438,  15042,  -5971,  11270, -12223,  13575,   9210,  -6635,
     -2556,  -7345,  -9309,  -8915,  15590,   5050,  -2567,  -8740,   3564,
    -15486,  13911,  -1418,   2263,  13794,    676, -15559,  -2040, -16327,
    -11587, -14457,  10238, -14641,  -5476,  16128,  -8120, -14706,  -4447,
     -7813,  11108, -15215,  -3160,  -2955,   3870,   2218,   1567,  -4383,
      1821,  -4332,  15152, -14393,   2255,   3921,    877,  -2888, -10104,
     -1013, -10116,   5015,  12168,   -351,    839,  -6652,   8605,  -5919,
      7855,   -698, -12814,  -9720,   8127,    -78, -15254, -13717, -14973,
    -16310,  -4217,   8267,  -2931,  11906,  10884,  -8447, -12175,  13234,
     -8066,  -3332,  -6990,  11170,  15185,   7957,   5903,  -5380,  -8237,
     -4057, -11634,   8561, -14799,  -4130,   6090, -15350,   7207, -10922,
     13906,  -4053,  -9901, -14018,   1074,  -3332,  -2582,   7863, -10919,
     -5786,  -5110,  -3696,  -5870,   6881,  -1301,  10302,   3577,  -7465,
    -14276,   1078, -12489,  -1898, -15027,    -20,   8139,  -4730,   7043,
    -14791, -14874, -15385,  -1856,    361, -15749,  -2705,  -9883,  -3702,
    -15919, -13885,   -417,  -9349,  -3323,   9553, -14954,  -3791,  11017,
      8756,   4349,    856, -15507,  15348,   7780, -12345,  13618,  -5501,
     15889,   2724,  -6553,    833, -13347,  11400,   4798,   3209,  -1986,
      3236,    747,  16354,   8891,  15639, -14773,  15017,  15015,   7645,
     13587,  -8479, -11656, -12921,   8821, -13941,   6962,  -6448,  16285,
    -14766,   6764,  -3942,   8277,  -7889,   9877,   4901, -10814,  13775,
    -11295,   -467,  13673,   5502,  11691,  13184,    880,  15808,  -7136,
      1903,   7769,  12144, -10085, -10376, -13137,    237,   3253,  -6006,
     -4712, -10154,  13879, -13630, -16087,  15998, -10849,  -4517,  -6381,
     13142,   8747,   8055,   3534,   7093,  -1876,    158, -12864,  -8949,
      5230, -15623,   5222,  12906,  12780, -14177,   8795,   1491,   4448,
      5939,   2282,  10833, -13601, -14380, -10962,   6838,  -4932, -13068,
    -12702,   8700,   3251,  -9768,  -5054, -14376, -13821,  15615,   7191,
     -9084,   2723,  -4658,  16161,   9826,  -8797,  -6755,   9421,  -3413,
     -3256, -11062,  -1844, -14253,   5861,  10480,   5744,  14899, -13186,
     -2645, -16073,   -354, -10518,   3159, -13993,  13216,   7472,   3913,
      7644, -12806, -16204,   4620,  -9280,   5640,  11275,  10487,   7882,
     -6171, -12633,    850, -11565,  -6894,   -263,   1694,  -8115,  -3417,
      3363,  -1063, -13495,  -6064,   6465, -12731,   8549,   8977, -10169,
     -3749,  -1669,  -3112,   2204,  15593,  12609,  12422,   7054, -11611,
     10139,   3738,  14052, -10340,   9065,  -2531, -12329, -16258,    526,
     -9463,  -8145,    993,  -5065, -15002,   4772, -15324,  -3562,  -4779,
      -525,  -8391,   2177, -15172, -11056,  15096,   8121,  -3035, -13789,
      6785, -11478,    267, -14547,   8728,  13750,  -1055, -11464,   2805,
     -1781,  -8048,  -6975, -13922, -13728,  11371, -11377,  -7623,  -8864,
      9227, -14372, -13263,  14496,  -4036,   1278, -10131,  14983,  -6473,
      9608,   5579, -15176,  10300,  -7724,  -4544,  11555,  14254,   1638,
     16011,   8210, -15001,   -201, -11943,   9697,  -3539,   7658, -13982,
     13869,  -1520, -11362,  16177,  -8830,  -5932,    777, -14153,   5173,
     13987,  15292,  13941,   6704,  -2420,  11696,   4411,  -4841,  13891,
    -16238, -14933,   5482,     80,  -2466,  14270,  -4713,  -1777,    152,
     -4460,  10129,  -9556,  -5409,  -6755,  -4545,  -5447, -10764, -12781,
     -6107,   5279, -11400, -13858,   4124,   7247,  10886,   3647,   5811,
      -582,  -5304, -16093,  -8004,  15611,  -1054,   6732,   1448, -11339,
      4807,  -4048,   9495,   6281, -10056,   5749,  -1630,  14756,   1460,
     -4011,   4986,  -2848,   9745,   5665,  -3951,  13007,  -9579,   -447,
      5742,    270,   4766,   4815, -11868, -12269,   -559,  -6665,  -9634,
     -8988,   9008, -11269,  -8316,   6098,  10886,   2086,  15057,  -7090,
      4346, -10457,  -4408,   3940,    495, -10156,    -44,  16126, -14103,
    -15754,  -7937,   5565,   6523,  -5222,  16355,  -1533,   9988, -10947,
     16001,   4095,  -5951,  12759, -12356,  -9439, -16268,  11807, -13282,
    -13766,  13277,   3887, -12418, -12315,  -3845,  13312,    -79, -13747,
      3277,   7889,  -4176,   3362,  -4605, -15434, -10724,   3191,  10448,
     -1639,  -5876,  -6962,  15978,  -2659,   3095, -16026,  10270,  -9947,
     -8560,   8875,  -5214,   4666,   3630,  -9296,  11481,  -5113,  -3757,
     -2735,   7994,  -7834,  -7939, -13737,   1598,   -190,   2250,  -9243,
    -13196, -10893,  13471,   6834,  -7035,  13560, -11076, -15265, -16210,
     10367,    760,  -9342,  10958,  15508,  -6041,  12283,  16195,  -2901,
     16077,  -8402,   9543,   5350, -15544,  -6818,   5235,   -658, -15706,
     -4600,   3872, -13557,  16144,   7003,  15771,   7412, -10365, -10747,
     -8306, -11444,   3141,   1642,  -5500,  -5914,  -1414,   6691, -14710,
     -7337,  -3451,  -1189,  -1622, -15983,  -3659,   2962,   1123,    587,
     10411, -12498,   4484,   2684,    783,   7407, -11398,  -1218, -15951,
      6123,  -1895,  12049,  -9672,  -2055, -13500,    785,  -5585, -13547,
    -13327,  -2984,   8244,   6553,   2135,  -8936, -10602,  10691, -15993,
     -9730,  15448,  -2865,   4362,  -2329,  -3649,   7722, -15092,  -9615,
     -2941,   6072, -11400, -10808,  12837,  -6553,  -3399,  -2166,   8234,
     -1137, -15634,  -2892,  13895,  16309,  13232, -10001,   9477,     77,
     13424,   9573,  -4970, -13043,  -8557,  13894, -10870,  -1194,   3432,
    -11171,  -1219,   8150,   8381,   7176,  13560,  -5348,   5206,  -7602,
     -5498,  -8580,  -2194,  -1701, -15194,  13496, -15453,  15294,   6716,
      8667,  11913,   2096,  15949,  -2783, -14370, -15621,  -4664,    837,
     -3034,  15061, -13566,  -1732,   1016,   -693, -14861,   3803,  -5395,
     -8204,   4105,    582,  -5992,  -8652,  -1240,  -9914, -14160,  14049,
     -8273,   3035,    902,    276,   3311,    428,  -1860,   9049,   6221,
      1059,  14283,  10768,  13030,  -6243,  -5393,  -6359,  12720,   4197,
     12663,   1390,  -2452, -13248,    775, -10598, -14446,   4897,  -7361,
     13681, -14804,   8878,  -3896,  -8663,  15297,  -1803,  -1859,  10882,
      7242, -12727,    169,   4151,   7141,    -59,  -2699,  10106, -14175,
        23,   4862, -11543,  15171, -14159,  13492,   -509,  15521, -11588,
      2575,   8489,  14792,  15154,   5934,   -653, -13821,  12252,  10117,
      -235,  -6182,   9080,   3531,   7984, -11268,  11350,   7552, -12245,
     15515,   6307,   8594,  -3284,  -7039,   9220,  15335,   7462,  -7480,
     10851, -14952,   8035,   8583,  -3827,   5606,  15217,  -7579,  -2082,
      -742,  -3241,  -8069,  11682,  -2942,   3882,  -1433,     -5,  -6316,
     -8061,  -1574, -10660,   9850,  -2749, -12762, -12704,  14987,  -8225,
     13658,   9163, -13370,  12141,  -7227,  13571,   8851,     99,   5811,
     12724,  11297,   6665,  14250,   2726,   2863,   9011,   8262,  11309,
       193,   3053,   2319, -15042,  12607,   1743,  13940, -11804,    485,
     -1199,   2202, -15765,  -4086, -13878,  -7580,  13200,  -1209,   2639,
     15499,   -148,   5256,  -3629,   3877,  -7544,   4101, -13771, -15985,
      5543,   -883,  15264,  -4664,  -6133,   5171,  -1746, -11160, -10507,
      4412,  -8177,  14364,   6319, -13272, -15950,  -2705,   2999,   4082,
    -15090, -14934, -11732,  10024,   6239,  -1029, -10791,   1878,  -7479,
     10289,   7681,  -1495,  -7077,   4256,  12458,  12550,  -9480,   5180,
      1333, -14314,  13225,   1521,  -8229,  -7478,  -3938,   6108,   7911,
     -7405,  10619,  -3991,   2857,  14156,  13006,  -6517, -14343, -15846,
    -12702, -15314, -10617,   1624,  12223, -12342,   3586, -15871,  -3128,
      7100,  10138,   -538,  15692,  -9701,  -5263,   6106,   8368,   2350,
      7922,  -3831, -15870,   5229, -10146,  15904,   3043,   1160,  15678,
        52,   5759, -10622,  11674,  -8353,  -5943,  15668,   3068,   4224,
     12424,  -9527,   7289,   8520, -13568,  13475,   5704,    -32,  -5198,
    -13033,  -9668,  14016,  15611,  -3941,   -142, -10369,  16001,   8912,
      4368,   6990,   8184,  -6694,  -3698,   9303,  -9780,  13763,  15711,
      8137, -14096, -11274,  -6318,     -6, -13321, -13758, -14632,   4677,
     -4095,  -7387,  -7800, -12853,   9281,   2648, -15205, -12857,   5187,
     15085,  13501,  -1018,  -1989,  -7821,  -7354,  -2053,  14734,   5859,
    -11527,  12165,  12865,  -4107,  -1771,  11896,  -7657,  -3553, -11039,
     11109,  -7095,  12047,  -8558,  -7504,   -192, -12697,   8352, -12862,
      3399,   1740, -12301, -15368,   -117,  15407,   7726,   5929,   6325,
     -2251,  11947, -14763,  11304,  12721,    -36, -16355,  -2954,  -5045,
      7275,  -8280,   8046,  13188,  -9177, -13833,   2228, -15848,  12776,
     -6456,   6289,  -2361,  15916, -10314,   6744,  -2877,   3893,  -8405,
     -4379,   5309,  14930,  15576,  -7304,   6427, -16272,  -3087,   7003,
     12579,  -1798,  13591, -10489,  -2440,   3255,    363,   8601,   -649,
    -14240,  -2106, -10512,  11219,   7972,  -4796,  11239,  -4433,  -6244,
    -11711,   6573,  14362,  -5925, -13163,  -4013, -10908, -12999,  11344,
      9364,  -1772,   6698,  -5048,  -5395,   -880, -11921, -15974,   3490,
      7310,  -3476,  15036,   4823,  11201,   7278,    303,   1673,  -2596,
     12342, -10963,   3290, -13573,   3704, -12878,   -325,   8118,  -6111,
    -13759,  -4733,   4900,  -7126, -14366,   9140,   6581,   9196,  14826,
      6462, -10106,   3039,  11296,   6111,  10544,  13792,   1340,  11061,
     15518,   6169, -13963,  -8572,   1421,  13198,  -4285,  -3806, -15303,
    -13202, -15391,   -466,  11501,  13171,   5646,  13221, -12147,  -7527,
     -4350, -13046,   3766,  -4385,  -3506,  -9397,   2380,     98,  -5112,
     -3902,  -7668,  -3072,   7681, -13995,   5704,   3240,   8868, -12944,
    -14654,  14599,   3589,  11223,   3731,  -9789,  -2051,    764, -10386,
      5062,  -1619, -13180,  -1771, -13167,   -286,  -2705,   2107,   2572,
    -12200,   -538,  13958,  13497,  15987,  -5824,  -7817,  11047,  -7066,
    -13704,  10496,  14030,   9532,   8916,  -2081,   8869,  -9184,  -2023,
     -7781,  -1355,    109,  15996, -14267,   4277,   5608, -12547, -15032,
     -6846, -15043,  -8994,  16000,  13907,   7515,   -851,  -4086,   8203,
    -12541, -11409,    818, -11991,   2286,  11940,   4441,  14166,  -3248,
     12670,  11136,   -243,   9391,  14842,  13983,   -728,  15923, -10342,
     13287,  -4484,   3692,  11657, -12138,  -4322,   -467,  11375,  -2624,
       820,  -8117,  -6519,  -9695,  -7998,   6108,   1797,  -9361, -15240,
     -7885,   7807,   3886,  -8439,   6767,  11724,  15378,  -3260,  -7101,
    -15302, -13125,  12372,  11113,  16311,   9915,  12052,  12567,  -4119,
     -8494,  -9816,  10046,   6675,  -3191,   -629,    164, -10175,  13005,
     -4728,   7597, -12608,  -3705,  -8764, -14929, -11095,   1491,  -2881,
     13286,   9655, -11452,  -5684,  -4256,  -8081,  13946,   9157,   5972,
      9524,  -2117,  -8125,  -2774,   4953, -11983,   2520,  -5501,   6517,
      4797,  -9245,  -7694,  -3644, -11649,  -5113,  10568,  -5182, -13000,
     12897,  -8913,  -7665,    998,  -8043,  -2889,  -9158,  12989,   -985,
      8264,  15106,  -4636,  15647,   1651,   -890,   7711,   5607, -13785,
     -7026,  14079,  -8339, -16291, -15243,  10866,  -9837,  -7965,    712,
     -8594,  -5745,  15111, -16263,  -1796,  -5130, -11649,   -192,  -5155,
      7133,   9029,  10344, -13174,  -3642,  11727,  -9436,      6,  13818,
     -6791,   9002,  12327,  -8989,  -7235,  -5226,  13803,  -5123, -11787,
    -12601,   1528,   5965,   2555, -16227,   3721,  13085,  -2290,   8139,
     -7092,  15672,    796, -12535,  12367,   2622,   7106,   7999,  -6817,
       690,   6878,   2345, -11381,  -7340,  -5169,   8904,   2523,   9257,
      1125, -12315,   9820,  -9496,   3891,  14327, -14771,  -4306,    348,
    -16318,  14810, -13693,   3278, -12436,   5893, -15750,  -6636,   9825,
    -11478,   9157,  13429,    556,  -3529,   1113,  -1680,  -9200,  14214,
     13060,  -2096,  -3436, -12570,  13037, -14704, -14580,  -4788,   8597,
    -11876, -14259,   5373,   6556,    509,  -3515, -14353,  -4251,  -2533,
     12498,   8981,   8701,  -8237,   4893,   8528,   7878,   8576,   3089,
     10161,   5671,  -1466,  -4342, -10445,   -472,   5762,  14912,  16318,
      2447,  13799,    156,  -9711,     72,  -7532,   5532,  16356, -11890,
     -8063,    260, -15723,   3803,  -3292,   2556, -14597, -12044,  -1387,
      5463,  10062,   7949,  -8194,  12572,   1712,  -9761,  -3933,   9430,
       945,  -5351,   -472,   2096,  -9072,  -7770,   7349,   6550,  -6872,
     12932,  15157,  -4927,  -4790, -11537,  12132,  11199, -15960,   8365,
      1734,   4599,    851,   8217,  -1032,  12364,   2897,  11584,   4196,
     -2558,    -13,   4105,   -668, -13377,   5290,   8931,  -2150,   3236,
     15545,  12951,   5633,   6663,   4356,   3227, -13330,  13238,  -5997,
      5570, -13933,   6350,  -6742, -12811,     66,   6053, -10482,  -4394,
     -4965,  13038, -11357,   1485,  -8103,  -5630,  -2638,   1985, -15252,
     14465,  -2897,  -5788,  10974,   8399,   7288, -16242,   -652,  10941,
     -7003,  -4260,  -8802,  11684,  13061,   6272,  -6673, -13385, -11496,
     14107,   4307,   8533,     46, -14795, -11882,  15259,  14727,   7822,
      1052, -11510,  16292, -15546,   -202,  -1404,  -9490, -15668,   5581,
     -9999,   2918, -10260,   6499,  -3956,   1935,  -2164, -10704,   2441,
     -7681,   4575,   7074,   8763,  -5660, -12460,  12371,  -4756,   4437,
     12622,   1054, -15514, -11392,  14816,   8346,   8846,   2703,   1008,
     -6202,   9191,   7089,  13336,   3254,  -2970,  -6456,  -7417,   4787,
     14025,  13609,  -6481,  -7567,  12630,  -1078,  -2114,  10115,  -3611,
      7433,   1790, -15491,    632,  16225, -11429,  -6154,   2886,  -5008,
    -13119,  -9640,  -3210,   8903,   6472,  -9726,  -4036,  -4686,   4723,
     -9927,  -7395,  -2393,  16027, -15438,  -9317,  -2776,  -6679,   2811,
      8182,   -101,   5785,  -3241,  -4865,  -6189,  -6611, -13795,  -7860,
      3840,   8087,     -6,  14432,    617,   -907,  -2729,   2619,  10385,
     12145,   6758,  11821,  -3316,   9102, -12833,  14930,   1642,  13637,
     -4609,  -2308,  -6094,   8419,   3372,   2443, -13720,    615,   6944,
     15190,   6630,  14119,   1239, -11272,   2096,  -9623,   4440,  -4912,
     -6561,  15453,  16378,   2331,  -5539,  16010,  -5675,  -1111,   4284,
     15651,  11166,   4362,  -2962,  -7653,  14112,   8615,  10706,  -8681,
    -14493,  -6740,    928,   1036,   3963,   1275,   6188,   5101,   9280,
     14822,   7138,   8218,  -2555,  -9270,  -4020,   6663,   9042, -11110,
      7088,  13950,    557, -12564,  12090,  -4104,  12472,   9793,    753,
     -5375,    352, -15027,   3206,   5491, -10489,  -7967,    595,   4565,
     12442,  11065,  11343,  -9557,   4872,  -9821,  -6917,   1584,  13939,
     13323,  -8709, -12439,   4067,  -7658,   7139,  -4806,   8700,   2591,
    -14259,  -6430,  10215,   2144,    262,  14361,   2758,   -401,  -7534,
    -14748,  -3808,  10204,  10813,  -9673,   3383,  13915, -15254, -16033,
    -11423, -10229,   9934,   4132,    559,   9188,  16209, -12610,  -5266,
      4377,   3315,   5300,   1035,   -305,  -1091, -11380, -12972, -11455,
     16127, -10495,  10010,   4860,  -7001,  -8370, -15116,  11892,  -1312,
      2458,   7016, -14063,  -7174,  15054,  10789, -15131,  -2261,  15347,
    -15015, -16106,   3560,  -1237,  14803,   3785,   2188, -16048, -12179,
     -4707,  -3094,  -5412,  -3217,  -5089, -16381,  -6244,   8490,  15256,
       838,  -3484,  -5195,  -4178,   -485,   6260,  14601, -10404,   2464,
      8245,  -7227,   1473,  12554,   7264,  -2193,  -8992,  15925,   4680,
    -10709,  -4800,  -9720,  -6632,  -7601,   8654,  -1021,   -173,  15988,
      3781,  -6564,   2085,  12321, -11243, -15100,  -8368,   1206,  15598,
    -11921,  -1465,   1347,  11084,  15377,  -2045, -13903, -11168,   9552,
    -15479,  12369,   7136,   8800,  11107,  -2456,   6732,   9105,  -1656,
     -6833, -11756, -13016, -14163, -13038, -11514,   1720,  12962,  13470,
    -13566,  15248,  -8815,   7769,   7298,  -9143,   5800,   9868,   2854,
     14575,  16248,   1696,   6924,   4306,    758, -11117,  -7196,  11526,
      4713,   6711,  12319,  -7018,   6509,   1889,   2092,   7553,  12382,
      5498, -13073, -14524,  -5026,  -4079,  -4123,  -1040,  -9219,  -7731,
      8361,   5464,  -1282,  13450,   3602,   4913,  12813,   5492,  -2506,
     13710,   1408,  -6649, -11446,  16099,  11383,   1463, -10191,  14905,
      -121,  -3561,  14131,  11000,   5026,  -1401,  -9385,  11518,    569,
      9187,  11610,  -2114,    444, -13040,  -5483, -12714, -15591,  14828,
       163,  -6082, -14323,  11027, -15954,  -1839,   1362,     14,  -5325,
     -8853,   8108,  -6701,  13811,   8375,  11005,   3340, -14709,   5751,
       710, -10844,   8243,   2643,   6906,  10642, -13788,   3957,   6852,
     -8664,   2389,  14392, -14143,   3130, -14854,   5967, -14400, -16119,
    -12970, -14367, -12703,   7845, -13226, -11892,    281,  11091,  -5681,
      7221, -10359,  12297,   2230,   2232,  -5243,   1337, -12012,   7990,
     -2451,   9673,   4152, -16363,   6392,   5586,   9867, -15120,   3798,
     10987,  13356,  10047,  15199,   9245,  14507,  10479,  14778, -14334,
     10787, -14822, -10452,   1727,  -4824,   6219,    -51,  -5585,  -1509,
    -12793,   2705,  -7823,  14496,  12407,  14303,  11468,   8150,   5141,
     -5077,  -2344,  -4344,  16200,  -6438,   2545,  12369, -16154, -10378,
     10315, -14318, -15772,  12072,  11028,  14022,  11077,  -4641,   8743,
      9166,  14546,  -6136,  -2860, -12017,  10038,  11081, -14604, -13363,
     10892,  15075, -15728,  12812, -13799,   -690,   5149,  16163,  11858,
    -11549,   -194,   1429, -15468,  13349, -10351,  -6401,   3660,  -5603,
      7155,   9329, -10732,  -7550,  14871,   7391,  -6507, -11518,   9476,
     -2044,   2901,  -7879,  -2311,  13117,  -6441, -12831,   2213,  10619,
     -2592,  -9457, -10450,   4084, -15529, -14925,   7733,   8507, -15448,
     -3302, -12696, -15346,   6346,   6587,  11311,  -7674,  -6934, -12437,
     -7483,   1795,  -9967,    -22,  -6498,   4455,  -6464,  -6041, -13723,
     -2239, -13395,   7924,  -3961,   6186, -13097,  14190,  -1584,   4991,
      9803,  -2171,  -7551, -15142,   8327,  -8246,  11662,   5167,   2182,
    -12588,   9402,  -1332,  -8949,  -9625, -14459, -11274,   5450,   4730,
     -1638,    552,    736,   3759,   5508,  -4029, -11709,   8823,  15807,
     -8700,   3525,  -2434,  -3917, -14929,  14918,   8418,  10058,  13706,
    -16115, -14596,   -917,  -9949, -16292,   4404,  -6283,   2200,   2099,
     15905,  10026,  13125,  12337, -16215, -14599,  -2064,  -9969,  -8505,
     13726, -10738,   3606,  10488,  16364, -15850,  11517,  -2561,  -4473,
      2722,   5051,  -8733,   6472,  -8220,   3545,   9995,  16338,   1194,
     16346,   1098,   4939,   5233,  10673,    641,   6407,  -4642,  -9214,
     -4976,   6016,  11109,   1416,  12344,  -6703,  -2132,   8370,   9088,
      3999,  -6538,  14449,   2935,  -8732, -10833, -16300,   2500,  13990,
      4970,   8872,   8472,  -7020, -10129,  -4756,  10207,   3745,   3444,
    -16043,  -6720,  -4746,  -2067,   3636,  -1795,   8193,  11259,  -8919,
      7192,    270,  -8032,   2321,  -7939,   9008,  -8071, -10637,  15319,
     -4436,  -2117, -12329, -11223,  -9614, -10007,  -1390,   7560,   6281,
      6246,  -5503, -11768,  -9220,  -6001,   9582,   5260, -13451,   6721,
      1412,  11443,   6330,   9508,  12788,   1975,  -9853,   9567,   3070,
     -1642,  -6158,  -3007,  -5268,  10418,  -4091,   8941,  11097, -11764,
      1163,  -8675,    895, -11053,  -5674, -16324,   3075, -13964,  -5896,
     -1559,  -5269, -10085,  -4117, -11879,     58,    425,  11481, -12693,
     -3150,  -7429,   4905,   4164,  -1392,   3650,  -7887,  14000, -12777,
       552, -11325,  -6447,   -100,  13635,    379,   -926, -13599,  -7104,
      5248, -11870,  -3524,  -4482,   -342,   2863,   9495,  10836,   9591,
      4671, -11392,  -4142,  13780,  -2515, -11043,  15685,   4802,  12465,
     -4000,  -6507,   5403, -13186,  -4491,   -179,   9505,  -2198,   2773,
     -3881,  -5612,  -9094,   2720,   7643,   2906,  14386,   8308,   6277,
     13246,  13877, -11286,  -8724,  -7845,  16116,  -1156,  10796,   1275,
     15074,   9163,  -1041,  -6405,   6488,  14171,  -4143, -12598, -12545,
     -1195,   -330,   8678, -11113,  -9480,  11402,  13091,   3927,   4709,
     16071,  -8851, -14627,   9330,   7972,   6140,  10372, -10235,   2866,
      5024,   7122,  -9507,  10692,   6397,   3909,   7241,  -6851,   9018,
    -14300,
};

static void initializeData() {
    memcpy_dma_ext(network.Layers[0].ifm.data, INPUTS_DATA, sizeof(INPUTS_DATA), sizeof(INPUTS_DATA), MEMCPY_WRITE);
}

static void dumpResults() {
    _DBGUART("Layer 0 outputs\r\n");
    CNN_printResult(&network.Layers[0].ofm);
}
