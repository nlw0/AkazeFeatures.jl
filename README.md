# AkazeFeatures.jl
AKAZE image features in Julia, porting the [original code](https://github.com/pablofdezalc/akaze/) by Pablo Fernandez Alcantarilla and Jesus Nuevo.

While this is a pure Julia project, the present code seeked to closely replicate the original implementation. For instance, many functions have the same name as the corresponding one in the original C++ project. It may be refactored in the future to make it more suitable to Julia, potentially enabling some specializations, etc. Our first objective was just to have a pure Julia implementation while making sure we produced very similar results.

The picture below is a plot where the Julia features are the dashed orange circles, and the original AKAZE extractor produced the blue circles underneath.

![demo](https://raw.githubusercontent.com/nlw0/AKAZE.jl/master/piratakaze.png)

## Quick tool demo

    user@N240WU:~/src/AkazeFeatures.jl$ ./akaze-features.jl images/pirate.png
    Extracted 938 points
    user@N240WU:~/src/AkazeFeatures.jl$ head keypoints.txt
    61
    938
	366.244962 29.033076 4.800000 1.699794 32 31 6 0 78 152 60 62 223 255 252 15 0 0 255 56 14 128 3 112 128 223 47 128 0 0 32 18 255 143 127 133 27 0 248 255 194 29 127 16 251 251 255 255 9 252 17 33 0 0 0 128 64 4 217 191 35 220 247 0 39
	342.703762 30.664175 4.800000 5.693416 33 6 126 1 46 16 8 128 109 3 244 8 132 0 68 218 255 15 1 224 0 255 244 39 0 2 0 28 0 0 10 2 128 32 249 247 127 11 128 111 145 212 74 90 240 229 147 200 0 26 64 1 4 0 169 218 1 253 43 116 10
	446.000648 36.506922 4.800000 5.119099 0 6 6 0 134 0 33 128 254 135 207 3 192 121 255 223 14 32 1 32 0 127 17 0 0 0 92 4 0 48 96 2 0 0 184 255 187 15 0 0 255 43 22 193 143 55 0 0 0 224 248 142 127 252 225 195 131 243 255 255 15
	447.826319 40.518104 4.800000 6.186299 33 22 108 0 128 13 56 128 233 6 164 157 7 0 254 207 157 32 6 192 0 14 112 235 1 0 0 188 3 208 126 68 0 1 248 235 171 30 49 0 170 171 213 129 111 207 33 126 128 11 0 0 128 120 226 229 231 227 252 255 11
	94.402266 44.270693 4.800000 0.073579 244 255 3 167 233 255 190 254 223 255 55 236 247 233 255 26 0 29 64 15 232 128 14 244 240 161 227 255 255 239 63 200 239 255 111 254 230 63 255 252 185 248 255 151 34 3 225 125 248 15 239 240 222 127 255 255 255 31 6 0 39
	99.040592 47.940726 4.800000 3.540063 128 246 7 0 0 96 67 11 229 47 18 128 195 113 254 255 0 0 0 0 0 5 0 128 0 0 0 224 120 255 1 36 3 203 96 57 247 191 126 18 100 252 19 254 3 0 32 60 152 7 243 48 255 249 231 205 207 131 121 255 63
	450.486521 49.357134 5.708194 0.100474 32 126 100 2 254 15 56 192 127 7 122 143 7 0 254 207 28 23 3 224 128 255 127 255 0 0 0 252 253 191 53 4 128 17 124 255 231 31 0 0 88 255 255 1 252 254 49 126 4 0 0 8 192 248 255 255 3 224 121 255 53
	287.575274 58.457806 5.708194 0.995287 103 61 236 55 137 159 240 46 211 252 73 31 129 32 244 203 189 127 231 207 136 136 240 254 121 35 0 248 255 9 244 221 115 247 36 136 174 121 111 16 33 42 254 197 9 252 63 44 14 195 97 31 230 176 195 0 0 192 176 255 3

## Bibliography
Some interesting scientific papers related to AKAZE:

 - Pablo Fernández Alcantarilla, Jesús Nuevo, and Adrien Bartoli. "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces", BMVC 2013.
 - Pablo Fernández Alcantarilla, Adrien Bartoli and Andrew J. Davison. "KAZE features.", ECCV 2012.
 - Xin Yang and Kwang-Ting Cheng. "LDB: An ultra-fast feature for scalable augmented reality on mobile devices." ISMAR 2012.
