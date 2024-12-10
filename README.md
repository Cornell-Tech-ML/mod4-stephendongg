# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


# Training Logs (4.5)
Training logs for quesiton 4.5!Please click on the arrow to expand for the trianing logs.



Sentiment (SST2)
<details>
  <summary>Click here to expand for Training Logs</summary>
  <pre>
Epoch 1, loss 31.759761223566635, train accuracy: 47.78%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 2, loss 31.242119451987875, train accuracy: 52.67%
Validation accuracy: 55.00%
Best Valid accuracy: 55.00%
Epoch 3, loss 31.06597703460005, train accuracy: 52.89%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 4, loss 31.2310148923931, train accuracy: 51.78%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 5, loss 30.83768209938271, train accuracy: 56.89%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 6, loss 30.61846812990396, train accuracy: 58.44%
Validation accuracy: 65.00%
Best Valid accuracy: 65.00%
Epoch 7, loss 30.319677006669938, train accuracy: 59.33%
Validation accuracy: 64.00%
Best Valid accuracy: 65.00%
Epoch 8, loss 30.57893005718152, train accuracy: 56.00%
Validation accuracy: 64.00%
Best Valid accuracy: 65.00%
Epoch 9, loss 29.817679988007384, train accuracy: 65.56%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 10, loss 29.480972297308732, train accuracy: 65.56%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 11, loss 29.443555635670823, train accuracy: 66.44%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 12, loss 29.349745259100654, train accuracy: 66.00%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 13, loss 28.678785485811986, train accuracy: 66.22%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 14, loss 28.246139749114434, train accuracy: 70.22%
Validation accuracy: 69.00%
Best Valid accuracy: 72.00%
Epoch 15, loss 27.674235772564597, train accuracy: 71.56%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 16, loss 27.18232955373037, train accuracy: 73.78%
Validation accuracy: 71.00%
Best Valid accuracy: 72.00%
Epoch 17, loss 26.970972010746564, train accuracy: 72.22%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 18, loss 26.42538942396672, train accuracy: 72.22%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 19, loss 25.325802022576337, train accuracy: 76.89%
Validation accuracy: 72.00%
Best Valid accuracy: 73.00%
Epoch 20, loss 25.7359600061219, train accuracy: 76.22%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 21, loss 24.016939575620935, train accuracy: 78.22%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 22, loss 23.711111478459657, train accuracy: 77.56%
Validation accuracy: 76.00%
Best Valid accuracy: 77.00%
Epoch 23, loss 22.8050626864828, train accuracy: 78.67%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 24, loss 22.033262879575677, train accuracy: 80.00%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 25, loss 21.556599661494655, train accuracy: 80.44%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 26, loss 21.166436998137574, train accuracy: 80.22%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 27, loss 21.001644175433697, train accuracy: 80.67%
Validation accuracy: 76.00%
Best Valid accuracy: 77.00%
Epoch 28, loss 19.52785112185904, train accuracy: 82.89%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 29, loss 18.84840230926969, train accuracy: 84.89%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 30, loss 18.53287363943704, train accuracy: 83.56%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 31, loss 17.564236295136542, train accuracy: 85.78%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 32, loss 17.64723887821848, train accuracy: 84.89%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 33, loss 16.32531332404229, train accuracy: 89.33%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 34, loss 16.85384180689128, train accuracy: 84.89%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 35, loss 15.802324420812043, train accuracy: 87.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 36, loss 15.164620254758345, train accuracy: 88.67%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 37, loss 14.113036284829274, train accuracy: 89.56%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 38, loss 13.53954273896855, train accuracy: 91.11%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 39, loss 13.648318065069569, train accuracy: 89.11%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 40, loss 12.914831820789109, train accuracy: 92.44%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 41, loss 12.930762603139625, train accuracy: 90.44%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 42, loss 11.95016346707527, train accuracy: 92.89%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 43, loss 11.794800633787613, train accuracy: 92.00%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 44, loss 11.690352891495918, train accuracy: 92.89%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 45, loss 12.043506498647629, train accuracy: 91.11%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 46, loss 11.203914806372197, train accuracy: 92.44%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 47, loss 10.959535691797102, train accuracy: 93.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 48, loss 10.439814347761413, train accuracy: 93.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 49, loss 9.797964675889357, train accuracy: 95.11%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 50, loss 9.487479231999272, train accuracy: 94.67%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 51, loss 9.061436591832978, train accuracy: 94.67%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 52, loss 9.36036470184215, train accuracy: 94.00%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 53, loss 8.500883543618551, train accuracy: 96.22%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 54, loss 9.622287899327864, train accuracy: 94.67%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 55, loss 8.421819842931232, train accuracy: 95.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 56, loss 8.072439824756207, train accuracy: 95.78%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 57, loss 7.952870611118277, train accuracy: 96.67%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 58, loss 7.512749589889968, train accuracy: 96.44%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 59, loss 7.506213832788808, train accuracy: 96.22%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 60, loss 7.14099792370223, train accuracy: 97.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 61, loss 6.949969627523928, train accuracy: 96.44%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 62, loss 6.886692826297763, train accuracy: 98.00%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 63, loss 6.490099234002835, train accuracy: 97.11%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 64, loss 6.233416420233071, train accuracy: 97.11%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 65, loss 6.550617678568237, train accuracy: 97.78%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 66, loss 6.0136145414604165, train accuracy: 98.22%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 67, loss 6.211622085319078, train accuracy: 97.56%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 68, loss 6.144886846564598, train accuracy: 97.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 69, loss 5.592891924222228, train accuracy: 98.00%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 70, loss 5.574096316340271, train accuracy: 98.67%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 71, loss 5.828029924409681, train accuracy: 97.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 72, loss 5.383119509334681, train accuracy: 98.22%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 73, loss 5.796035468407915, train accuracy: 97.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 74, loss 5.238249674565037, train accuracy: 98.00%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 75, loss 5.125150925051718, train accuracy: 98.22%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 76, loss 5.455386536713055, train accuracy: 96.67%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 77, loss 4.8361144504479014, train accuracy: 98.00%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 78, loss 5.313373942226967, train accuracy: 98.00%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 79, loss 4.810822028377119, train accuracy: 98.44%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 80, loss 4.292718322473639, train accuracy: 98.67%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 81, loss 3.971767292023638, train accuracy: 98.89%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 82, loss 4.404646762264847, train accuracy: 98.44%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 83, loss 4.361882426161569, train accuracy: 98.44%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 84, loss 4.012673142863262, train accuracy: 99.56%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 85, loss 4.057351334261369, train accuracy: 99.11%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 86, loss 4.626264974674467, train accuracy: 98.00%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 87, loss 4.230502329325194, train accuracy: 98.89%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 88, loss 3.8698089454556155, train accuracy: 99.11%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 89, loss 3.8436573645820595, train accuracy: 98.67%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 90, loss 3.892983669225886, train accuracy: 99.11%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 91, loss 3.7294353517600762, train accuracy: 98.44%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 92, loss 3.7881248339970797, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 93, loss 3.317668309466243, train accuracy: 99.33%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 94, loss 3.380099565276426, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 95, loss 3.6568403826102847, train accuracy: 98.89%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 96, loss 3.402991820862824, train accuracy: 99.11%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 97, loss 3.33380087306467, train accuracy: 99.11%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 98, loss 3.4641092665968007, train accuracy: 98.67%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 99, loss 2.9305531577410893, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 100, loss 2.973645237411781, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 101, loss 2.9926164701212934, train accuracy: 99.11%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 102, loss 2.9530412853301775, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 103, loss 3.092348525918887, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 104, loss 2.837920133026399, train accuracy: 98.89%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 105, loss 2.663110359045307, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 106, loss 3.0076907632455394, train accuracy: 99.11%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 107, loss 3.0212331762800804, train accuracy: 99.11%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 108, loss 2.8863687240934306, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 109, loss 2.7931721188958707, train accuracy: 98.89%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 110, loss 2.7370313642522537, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 111, loss 2.596685060110418, train accuracy: 98.89%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 112, loss 2.496650952827238, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 113, loss 2.3066411228493315, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 114, loss 2.456275541632487, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 115, loss 2.202316872015371, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 116, loss 2.097013592181875, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 117, loss 2.4773788643387755, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 118, loss 2.7179296389770253, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 119, loss 2.1051702474204204, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 120, loss 2.2119135081177417, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 121, loss 2.425497912876913, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 122, loss 2.1955602327322827, train accuracy: 99.78%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 123, loss 2.330927512275077, train accuracy: 99.11%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 124, loss 2.395036416326893, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 125, loss 2.246457952385793, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 126, loss 1.9221928054981294, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 127, loss 2.2694398398446394, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 128, loss 2.0982288636935627, train accuracy: 99.11%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 129, loss 1.9901324812634031, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 130, loss 2.0835426102025636, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 131, loss 1.9895113878439934, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 132, loss 1.9239343989529802, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 133, loss 1.9868919697193264, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 134, loss 1.735814362245653, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 135, loss 1.5691837542459386, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 136, loss 1.828521297455666, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 137, loss 1.9353039701966137, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 138, loss 1.645793644825693, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 139, loss 2.0819897145640898, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 140, loss 1.821853229639308, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 141, loss 1.7734693157780126, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 142, loss 1.9135455552641385, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 143, loss 1.7006703600293536, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 144, loss 1.7396675447704248, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 145, loss 1.6837446250129182, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 146, loss 1.6860208190264105, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 147, loss 1.4514379372700992, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 148, loss 1.8828709557857535, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 149, loss 1.58534474036673, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 150, loss 1.652507828117553, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 151, loss 1.4341309645403697, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 152, loss 1.6994062253558773, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 153, loss 1.6800983789932915, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 154, loss 1.4749830001721773, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 155, loss 1.766253084532715, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 156, loss 1.7536470563713216, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 157, loss 1.3957987243403027, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 158, loss 1.6702027276957547, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 159, loss 1.548705606428437, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 160, loss 1.6772645464626226, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 161, loss 1.5450219642567558, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 162, loss 1.3368628246261658, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 163, loss 1.8302908790397874, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 164, loss 1.601587547229175, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 165, loss 1.459976886355088, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 166, loss 1.4319367408465624, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 167, loss 1.7146553235348334, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 168, loss 1.4897537628515225, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 169, loss 1.3415030981125793, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 170, loss 1.3806160518547421, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 171, loss 1.284222606211177, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 172, loss 1.17577878042559, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 173, loss 1.2274480045021738, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 174, loss 1.464929051178964, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 175, loss 1.192216697958693, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 176, loss 1.3502198499158011, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 177, loss 1.0630374375466256, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 178, loss 1.3610819354361876, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 179, loss 1.3811309007423833, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 180, loss 1.4509984681879418, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 181, loss 1.0604928835338698, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 182, loss 1.3557078378104048, train accuracy: 99.33%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 183, loss 1.4151847961306232, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 184, loss 1.1696818846231365, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 185, loss 1.4040880482056401, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 186, loss 1.2925817810886877, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 187, loss 1.164021877796226, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 188, loss 1.1382592788725592, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 189, loss 1.1586679594814915, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 190, loss 1.1143591013186853, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 191, loss 0.9795071026770814, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 192, loss 0.9695974949004961, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 193, loss 1.1071246172058395, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 194, loss 1.1704288028256744, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 195, loss 0.9973751463051131, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 196, loss 1.12952121302821, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 197, loss 1.0949633920299975, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 198, loss 1.2155477718414187, train accuracy: 99.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 199, loss 1.0943996056587801, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 200, loss 0.869147519783082, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 201, loss 1.115329641091719, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 202, loss 0.9747610942854046, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 203, loss 1.026509816852287, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 204, loss 1.0833430860464337, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 205, loss 0.8269161376879643, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 79.00%
Epoch 206, loss 1.1043379783868292, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 207, loss 1.0816121890598824, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 208, loss 0.965512538034573, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 79.00%
Epoch 209, loss 0.9410415797367871, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 210, loss 1.2143445156724082, train accuracy: 99.11%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 211, loss 1.0619022916986167, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 212, loss 1.1066492359937288, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 213, loss 1.020948808198861, train accuracy: 99.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 214, loss 0.7855069899479977, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 215, loss 0.8592584063761173, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 216, loss 1.093931111336333, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 217, loss 1.0015070779191622, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 79.00%
Epoch 218, loss 0.8902340276335245, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 219, loss 0.8826713669941868, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 220, loss 0.8524770581599097, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 221, loss 0.9757472131734666, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 222, loss 0.9287945889940421, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 223, loss 1.0674561851315867, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 224, loss 1.2008841877651304, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 225, loss 0.918327063504677, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 79.00%
Epoch 226, loss 0.9542503662915235, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 227, loss 1.0897545470810626, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 228, loss 1.0399221440668351, train accuracy: 99.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 229, loss 0.9301479663401849, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 230, loss 0.8104795994823444, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 231, loss 0.8316053596233686, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 232, loss 1.0479518761326663, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 233, loss 1.0156482125225121, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 234, loss 0.7593913222497812, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 235, loss 0.866520790821926, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 236, loss 0.8149352488589882, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 237, loss 0.7262988664750518, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 238, loss 0.8263337845586043, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 239, loss 0.7479121620951826, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 240, loss 0.8353982213075863, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 241, loss 0.8461756706552044, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 242, loss 0.8809705725069867, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 243, loss 0.7518165533941055, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 79.00%
Epoch 244, loss 0.7106215241180295, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 245, loss 0.9648914586572355, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 246, loss 0.9238033376594412, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 247, loss 0.8392391046457546, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 248, loss 0.7678688570775086, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 79.00%
Epoch 249, loss 0.8802106902525269, train accuracy: 99.78%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 250, loss 0.9476368742883305, train accuracy: 99.56%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
</pre>
</details>
</pre>
</details>


MNIST Dataset
<details>
  <summary>Click here to expand for Training Logs</summary>
  <pre>
Epoch 1 loss 2.288567429958137 train acc 1/16 valid acc 3/16
Epoch 1 loss 11.467422907915527 train acc 4/16 valid acc 5/16
Epoch 1 loss 11.439674007626994 train acc 3/16 valid acc 6/16
Epoch 1 loss 11.09740879146991 train acc 6/16 valid acc 6/16
Epoch 1 loss 10.991821842054746 train acc 2/16 valid acc 7/16
Epoch 1 loss 9.946788815446904 train acc 5/16 valid acc 7/16
Epoch 1 loss 8.16578610103343 train acc 4/16 valid acc 10/16
Epoch 1 loss 9.156633776242113 train acc 9/16 valid acc 11/16
Epoch 1 loss 8.9810311888441 train acc 4/16 valid acc 11/16
Epoch 1 loss 7.938290437030163 train acc 7/16 valid acc 10/16
Epoch 1 loss 7.0235938351775165 train acc 7/16 valid acc 11/16
Epoch 1 loss 7.018824027624112 train acc 11/16 valid acc 9/16
Epoch 1 loss 7.3017963534156 train acc 7/16 valid acc 10/16
Epoch 1 loss 6.441387914570037 train acc 12/16 valid acc 10/16
Epoch 1 loss 6.223438530406455 train acc 7/16 valid acc 8/16
Epoch 1 loss 6.6387921805240655 train acc 7/16 valid acc 10/16
Epoch 1 loss 7.499738134844056 train acc 8/16 valid acc 12/16
Epoch 1 loss 5.600379024329033 train acc 10/16 valid acc 11/16
Epoch 1 loss 6.147891521488157 train acc 9/16 valid acc 12/16
Epoch 1 loss 4.346002752739345 train acc 12/16 valid acc 10/16
Epoch 1 loss 5.595732270812324 train acc 10/16 valid acc 10/16
Epoch 1 loss 3.8767120697636064 train acc 11/16 valid acc 14/16
Epoch 1 loss 2.7801294327500425 train acc 13/16 valid acc 10/16
Epoch 1 loss 3.6452303169863063 train acc 9/16 valid acc 11/16
Epoch 1 loss 3.380978264381279 train acc 13/16 valid acc 11/16
Epoch 1 loss 4.599292335905957 train acc 10/16 valid acc 11/16
Epoch 1 loss 4.561436669823843 train acc 11/16 valid acc 13/16
Epoch 1 loss 2.5316854127002606 train acc 15/16 valid acc 15/16
Epoch 1 loss 3.0171718367458307 train acc 12/16 valid acc 14/16
Epoch 1 loss 2.96511627603516 train acc 14/16 valid acc 12/16
Epoch 1 loss 3.3109875030082567 train acc 10/16 valid acc 13/16
Epoch 1 loss 2.678669222875408 train acc 15/16 valid acc 12/16
Epoch 1 loss 3.2139348564225605 train acc 12/16 valid acc 14/16
Epoch 1 loss 3.055208239580489 train acc 12/16 valid acc 13/16
Epoch 1 loss 5.650629522483795 train acc 10/16 valid acc 14/16
Epoch 1 loss 3.032536641169308 train acc 11/16 valid acc 14/16
Epoch 1 loss 3.145935836625177 train acc 11/16 valid acc 12/16
Epoch 1 loss 3.5170442701914615 train acc 11/16 valid acc 13/16
Epoch 1 loss 2.886444160662692 train acc 13/16 valid acc 14/16
Epoch 1 loss 2.9268458008026244 train acc 14/16 valid acc 14/16
Epoch 1 loss 2.202673094296144 train acc 13/16 valid acc 14/16
Epoch 1 loss 3.3109983120306894 train acc 14/16 valid acc 16/16
Epoch 1 loss 2.477848447520647 train acc 12/16 valid acc 14/16
Epoch 1 loss 2.4216863440050007 train acc 11/16 valid acc 12/16
Epoch 1 loss 3.2151935601481774 train acc 12/16 valid acc 15/16
Epoch 1 loss 2.349283246129232 train acc 13/16 valid acc 15/16
Epoch 1 loss 2.7733181921586882 train acc 13/16 valid acc 16/16
Epoch 1 loss 2.953968549959288 train acc 11/16 valid acc 14/16
Epoch 1 loss 2.348441860329396 train acc 14/16 valid acc 14/16
Epoch 1 loss 1.8949280608279016 train acc 16/16 valid acc 16/16
Epoch 1 loss 3.2056587264447773 train acc 14/16 valid acc 13/16
Epoch 1 loss 3.027831270949417 train acc 15/16 valid acc 14/16
Epoch 1 loss 2.432958628719133 train acc 14/16 valid acc 15/16
Epoch 1 loss 1.3232112561865599 train acc 16/16 valid acc 15/16
Epoch 1 loss 3.8585582045396007 train acc 13/16 valid acc 14/16
Epoch 1 loss 1.8996770547534143 train acc 14/16 valid acc 13/16
Epoch 1 loss 2.5334335293769383 train acc 14/16 valid acc 16/16
Epoch 1 loss 2.1115360598058954 train acc 14/16 valid acc 14/16
Epoch 1 loss 2.5658716490292686 train acc 13/16 valid acc 15/16
Epoch 1 loss 2.343714995493184 train acc 15/16 valid acc 14/16
Epoch 1 loss 3.803872754346108 train acc 13/16 valid acc 13/16
Epoch 1 loss 2.58235816915877 train acc 13/16 valid acc 15/16
Epoch 1 loss 2.3582715104199967 train acc 15/16 valid acc 15/16
Epoch 2 loss 0.2772318295060284 train acc 14/16 valid acc 15/16
Epoch 2 loss 2.1630733781781517 train acc 15/16 valid acc 16/16
Epoch 2 loss 3.1583484079987634 train acc 11/16 valid acc 15/16
Epoch 2 loss 1.6166735852048504 train acc 16/16 valid acc 14/16
Epoch 2 loss 1.5885973507812474 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.6077066748384803 train acc 14/16 valid acc 15/16
Epoch 2 loss 1.9465594946543132 train acc 11/16 valid acc 15/16
Epoch 2 loss 2.2473154460116307 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.590724851222562 train acc 16/16 valid acc 16/16
Epoch 2 loss 1.070227567831456 train acc 16/16 valid acc 15/16
Epoch 2 loss 2.9438420893359605 train acc 14/16 valid acc 16/16
Epoch 2 loss 3.2142083725048387 train acc 12/16 valid acc 15/16
Epoch 2 loss 3.0660491994454864 train acc 13/16 valid acc 15/16
Epoch 2 loss 2.25517928658542 train acc 11/16 valid acc 16/16
Epoch 2 loss 3.3940743588517917 train acc 11/16 valid acc 14/16
Epoch 2 loss 2.045915846139934 train acc 15/16 valid acc 15/16
Epoch 2 loss 2.668373259463841 train acc 14/16 valid acc 15/16
Epoch 2 loss 2.889837769518187 train acc 15/16 valid acc 16/16
Epoch 2 loss 1.8797583762509589 train acc 15/16 valid acc 15/16
Epoch 2 loss 0.9915256276879717 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.9603074503463427 train acc 12/16 valid acc 14/16
Epoch 2 loss 1.6313167451529975 train acc 14/16 valid acc 16/16
Epoch 2 loss 0.7591368741174441 train acc 14/16 valid acc 16/16
Epoch 2 loss 1.3499532200079376 train acc 15/16 valid acc 15/16
Epoch 2 loss 0.7486976304905681 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.2426718623937285 train acc 13/16 valid acc 15/16
Epoch 2 loss 0.6830276848352229 train acc 16/16 valid acc 15/16
Epoch 2 loss 0.7208011319302026 train acc 16/16 valid acc 15/16
Epoch 2 loss 2.1930765061794713 train acc 14/16 valid acc 15/16
Epoch 2 loss 0.8008269789491547 train acc 15/16 valid acc 14/16
Epoch 2 loss 3.1378087624656548 train acc 11/16 valid acc 14/16
Epoch 2 loss 1.9208307497137995 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.0614995710970745 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.712769521018692 train acc 14/16 valid acc 13/16
Epoch 2 loss 2.368688296918392 train acc 14/16 valid acc 15/16
Epoch 2 loss 1.6536950219345572 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.3360317175886376 train acc 14/16 valid acc 15/16
Epoch 2 loss 1.846687375463834 train acc 15/16 valid acc 14/16
Epoch 2 loss 2.0282637971654234 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.834374277531899 train acc 15/16 valid acc 14/16
Epoch 2 loss 1.0962321408864315 train acc 16/16 valid acc 14/16
Epoch 2 loss 2.2334933601532327 train acc 14/16 valid acc 14/16
Epoch 2 loss 0.8444933660657283 train acc 15/16 valid acc 16/16
Epoch 2 loss 1.256555454556251 train acc 15/16 valid acc 16/16
Epoch 2 loss 1.8556742556122994 train acc 15/16 valid acc 16/16
Epoch 2 loss 0.6850599936826339 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.3422139727857898 train acc 15/16 valid acc 14/16
Epoch 2 loss 2.205195634173831 train acc 14/16 valid acc 15/16
Epoch 2 loss 1.305871957171397 train acc 15/16 valid acc 16/16
Epoch 2 loss 1.098624499781831 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.5432038987357124 train acc 13/16 valid acc 15/16
Epoch 2 loss 1.7265541834684552 train acc 16/16 valid acc 14/16
Epoch 2 loss 1.1058365328217048 train acc 15/16 valid acc 15/16
Epoch 2 loss 1.0440715021116904 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.4899164622076602 train acc 13/16 valid acc 14/16
Epoch 2 loss 1.0217876097015317 train acc 15/16 valid acc 16/16
Epoch 2 loss 1.2253037310133152 train acc 16/16 valid acc 16/16
Epoch 2 loss 1.5379721681901815 train acc 15/16 valid acc 16/16
Epoch 2 loss 2.206138658028515 train acc 13/16 valid acc 14/16
Epoch 2 loss 1.746051058637762 train acc 16/16 valid acc 16/16
Epoch 2 loss 1.6632551400557243 train acc 14/16 valid acc 16/16
Epoch 2 loss 1.1204384028011383 train acc 16/16 valid acc 15/16
Epoch 2 loss 1.4847019840081421 train acc 14/16 valid acc 15/16
Epoch 3 loss 0.13313512527114585 train acc 16/16 valid acc 16/16
Epoch 3 loss 1.1902052974196908 train acc 15/16 valid acc 16/16
Epoch 3 loss 2.1437951976643843 train acc 13/16 valid acc 14/16
Epoch 3 loss 1.555812880468093 train acc 14/16 valid acc 14/16
Epoch 3 loss 0.8657738561772417 train acc 16/16 valid acc 14/16
Epoch 3 loss 0.9766122287549133 train acc 14/16 valid acc 14/16
Epoch 3 loss 1.4238081247799625 train acc 13/16 valid acc 15/16
Epoch 3 loss 1.458658474461804 train acc 16/16 valid acc 15/16
Epoch 3 loss 1.3490784156395337 train acc 16/16 valid acc 16/16
Epoch 3 loss 0.6580424804694871 train acc 15/16 valid acc 16/16
Epoch 3 loss 0.5146765584254029 train acc 15/16 valid acc 14/16
Epoch 3 loss 1.680704670558633 train acc 14/16 valid acc 15/16
Epoch 3 loss 2.6460970339473384 train acc 14/16 valid acc 15/16
Epoch 3 loss 1.3930794706772267 train acc 14/16 valid acc 14/16
Epoch 3 loss 2.283381697602047 train acc 14/16 valid acc 15/16
Epoch 3 loss 1.1743604275554 train acc 15/16 valid acc 14/16
Epoch 3 loss 2.010843221382218 train acc 14/16 valid acc 14/16
Epoch 3 loss 1.8993779892784421 train acc 12/16 valid acc 14/16
Epoch 3 loss 1.4523799779893367 train acc 15/16 valid acc 15/16
Epoch 3 loss 1.2718567194447685 train acc 14/16 valid acc 14/16
Epoch 3 loss 0.9672012287956322 train acc 15/16 valid acc 14/16
Epoch 3 loss 1.0095877991822368 train acc 16/16 valid acc 15/16
Epoch 3 loss 0.3749020739206847 train acc 15/16 valid acc 14/16
Epoch 3 loss 1.8296697775443307 train acc 15/16 valid acc 14/16
Epoch 3 loss 0.5520547058630624 train acc 16/16 valid acc 15/16
Epoch 3 loss 0.845776622847636 train acc 16/16 valid acc 14/16
Epoch 3 loss 0.8335448629268873 train acc 16/16 valid acc 15/16
Epoch 3 loss 1.0063677752040932 train acc 16/16 valid acc 15/16
Epoch 3 loss 0.8531279261293552 train acc 14/16 valid acc 14/16
Epoch 3 loss 0.5830994362739366 train acc 16/16 valid acc 14/16
Epoch 3 loss 1.4778269829984256 train acc 15/16 valid acc 15/16
Epoch 3 loss 0.9072359459585946 train acc 14/16 valid acc 15/16
Epoch 3 loss 0.7215071630760309 train acc 16/16 valid acc 14/16
Epoch 3 loss 1.7249046445027927 train acc 14/16 valid acc 15/16
Epoch 3 loss 1.3125654115659877 train acc 15/16 valid acc 16/16
Epoch 3 loss 0.7726480082523121 train acc 15/16 valid acc 16/16
Epoch 3 loss 1.4450436016117651 train acc 15/16 valid acc 15/16
Epoch 3 loss 0.7171601034803285 train acc 14/16 valid acc 15/16
Epoch 3 loss 1.4414710830038369 train acc 15/16 valid acc 15/16
Epoch 3 loss 1.3513891301404504 train acc 14/16 valid acc 15/16
Epoch 3 loss 0.7510480308391597 train acc 15/16 valid acc 15/16
Epoch 3 loss 1.1990279172063747 train acc 15/16 valid acc 14/16
Epoch 3 loss 0.6119108740424307 train acc 14/16 valid acc 15/16
Epoch 3 loss 0.5551073289713893 train acc 16/16 valid acc 15/16
Epoch 3 loss 1.84187359946634 train acc 14/16 valid acc 16/16
Epoch 3 loss 0.5576789661956412 train acc 16/16 valid acc 16/16
Epoch 3 loss 0.4461034356848175 train acc 16/16 valid acc 16/16
Epoch 3 loss 1.539151685534431 train acc 14/16 valid acc 15/16
Epoch 3 loss 1.3159623395754179 train acc 14/16 valid acc 15/16
Epoch 3 loss 0.9180581455910116 train acc 15/16 valid acc 16/16
Epoch 3 loss 1.0867093992695904 train acc 15/16 valid acc 15/16
Epoch 3 loss 1.200900799361937 train acc 16/16 valid acc 16/16
Epoch 3 loss 0.9139968462414945 train acc 16/16 valid acc 15/16
Epoch 3 loss 0.5190533787693768 train acc 16/16 valid acc 16/16
Epoch 3 loss 1.4929643389873555 train acc 14/16 valid acc 13/16
Epoch 3 loss 0.9483540923994311 train acc 15/16 valid acc 15/16
Epoch 3 loss 0.6272050988161573 train acc 16/16 valid acc 15/16
Epoch 3 loss 1.3304120162240518 train acc 16/16 valid acc 15/16
Epoch 3 loss 1.157081334342123 train acc 14/16 valid acc 16/16
Epoch 3 loss 1.9077337515500665 train acc 15/16 valid acc 14/16
Epoch 3 loss 1.4375607343647168 train acc 16/16 valid acc 16/16
Epoch 3 loss 1.0021197742489203 train acc 15/16 valid acc 16/16
Epoch 3 loss 1.61624478987776 train acc 14/16 valid acc 16/16
Epoch 4 loss 0.0171679163562084 train acc 16/16 valid acc 16/16
Epoch 4 loss 1.2218953145598839 train acc 15/16 valid acc 16/16
Epoch 4 loss 1.527914099559867 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.1545083127990021 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.2547383498782344 train acc 16/16 valid acc 14/16
Epoch 4 loss 0.7805761818621588 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.7888430330668186 train acc 16/16 valid acc 16/16
Epoch 4 loss 1.2402233678579284 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.9547484946133843 train acc 16/16 valid acc 15/16
Epoch 4 loss 0.5639366006587796 train acc 16/16 valid acc 15/16
Epoch 4 loss 0.60356129637179 train acc 16/16 valid acc 16/16
Epoch 4 loss 1.070579126625899 train acc 14/16 valid acc 15/16
Epoch 4 loss 2.06264698653883 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.9028763143539926 train acc 16/16 valid acc 15/16
Epoch 4 loss 1.7397114125455044 train acc 14/16 valid acc 16/16
Epoch 4 loss 0.8327594618035061 train acc 15/16 valid acc 16/16
Epoch 4 loss 1.260050295982615 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.8423278006593574 train acc 14/16 valid acc 16/16
Epoch 4 loss 1.0087142365725135 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.786302357036599 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.6762232138025883 train acc 15/16 valid acc 14/16
Epoch 4 loss 1.045077420130905 train acc 16/16 valid acc 15/16
Epoch 4 loss 0.2966191662830888 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.0455807675244269 train acc 14/16 valid acc 16/16
Epoch 4 loss 0.7655426829757557 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.1826976340472197 train acc 16/16 valid acc 15/16
Epoch 4 loss 0.8698594180857326 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.6138317407840286 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.7103995301405744 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.26158441321607573 train acc 16/16 valid acc 15/16
Epoch 4 loss 0.5914810261270683 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.6684885336701285 train acc 14/16 valid acc 14/16
Epoch 4 loss 0.24005634957716315 train acc 16/16 valid acc 14/16
Epoch 4 loss 0.564096736794665 train acc 16/16 valid acc 15/16
Epoch 4 loss 1.7894194493531583 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.8470509907163604 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.6434328166168815 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.8408882292648673 train acc 15/16 valid acc 16/16
Epoch 4 loss 1.3288065406871727 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.8048442359330874 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.26483615051639026 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.9909016344590929 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.6094366277788426 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.1365585702168286 train acc 13/16 valid acc 16/16
Epoch 4 loss 0.921404930710898 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.3260402829557467 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.6422104785075647 train acc 14/16 valid acc 16/16
Epoch 4 loss 1.7997198089432953 train acc 13/16 valid acc 16/16
Epoch 4 loss 0.8878692995197519 train acc 16/16 valid acc 15/16
Epoch 4 loss 1.0213508747407736 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.9084747276634995 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.269282519267374 train acc 14/16 valid acc 15/16
Epoch 4 loss 1.0992484265816818 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.41443012079547414 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.7646593116425718 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.5904108674430808 train acc 15/16 valid acc 15/16
Epoch 4 loss 0.7984434213376616 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.9888370071187926 train acc 16/16 valid acc 16/16
Epoch 4 loss 0.9425093423039117 train acc 15/16 valid acc 16/16
Epoch 4 loss 1.6000661498382676 train acc 15/16 valid acc 15/16
Epoch 4 loss 1.387940075228028 train acc 15/16 valid acc 16/16
Epoch 4 loss 0.7603830227895048 train acc 15/16 valid acc 16/16
Epoch 4 loss 1.0582123031725261 train acc 15/16 valid acc 15/16
Epoch 5 loss 0.01997055874157614 train acc 16/16 valid acc 16/16
Epoch 5 loss 1.0525521699956624 train acc 15/16 valid acc 15/16
Epoch 5 loss 1.2889873062982526 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.5089541892605373 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.40982747565968464 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.6761948759184274 train acc 15/16 valid acc 15/16
Epoch 5 loss 0.9265523697706409 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.8402000433277431 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.5417645971092192 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.12509553609433208 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.37205704888534036 train acc 16/16 valid acc 16/16
Epoch 5 loss 1.8246243971835165 train acc 16/16 valid acc 15/16
Epoch 5 loss 1.255741525291074 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.8443275886136565 train acc 14/16 valid acc 16/16
Epoch 5 loss 1.51016496356454 train acc 13/16 valid acc 16/16
Epoch 5 loss 1.188661108708498 train acc 14/16 valid acc 14/16
Epoch 5 loss 0.9196588478263678 train acc 16/16 valid acc 16/16
Epoch 5 loss 1.1626199377065622 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.7591472379865538 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.5957741060948435 train acc 14/16 valid acc 16/16
Epoch 5 loss 0.8946960441295047 train acc 15/16 valid acc 14/16
Epoch 5 loss 0.8807398640714184 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.42946003608330485 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.5000875260107943 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.5627128544185132 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.6549912519779577 train acc 14/16 valid acc 16/16
Epoch 5 loss 1.328018768456731 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.8355292384395806 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.6564520775547034 train acc 14/16 valid acc 16/16
Epoch 5 loss 0.3245119988732458 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.302583504967861 train acc 15/16 valid acc 15/16
Epoch 5 loss 0.33294973582365267 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.6173998935982765 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.8794479550357068 train acc 14/16 valid acc 15/16
Epoch 5 loss 1.7780346778509055 train acc 14/16 valid acc 16/16
Epoch 5 loss 0.4790379275382316 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.5281925386490528 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.679586885492082 train acc 15/16 valid acc 16/16
Epoch 5 loss 1.017911434335437 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.7431941211232407 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.44449054404941124 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.34021560290816055 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.4312133565250448 train acc 15/16 valid acc 16/16
Epoch 5 loss 0.6482133316841532 train acc 16/16 valid acc 16/16
Epoch 5 loss 1.2144407422836445 train acc 14/16 valid acc 15/16
Epoch 5 loss 0.31593826202525244 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.5357728948470529 train acc 15/16 valid acc 16/16
Epoch 5 loss 1.2186745526617984 train acc 14/16 valid acc 16/16
Epoch 5 loss 0.9155368634527803 train acc 15/16 valid acc 13/16
Epoch 5 loss 1.6897405291529273 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.8111156923719774 train acc 16/16 valid acc 16/16
Epoch 5 loss 1.1400963231589751 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.5365770430626412 train acc 15/16 valid acc 15/16
Epoch 5 loss 0.7287766523439296 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.654017604303117 train acc 15/16 valid acc 15/16
Epoch 5 loss 0.7509613267111533 train acc 14/16 valid acc 16/16
Epoch 5 loss 0.7239205648907612 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.6280741478933174 train acc 16/16 valid acc 16/16
Epoch 5 loss 0.9389924915097778 train acc 15/16 valid acc 16/16
Epoch 5 loss 1.233399699521902 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.7654937154908836 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.6585123042816322 train acc 16/16 valid acc 15/16
Epoch 5 loss 0.7150758611049102 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.20690548189674807 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.6449035201793787 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.0165432233906124 train acc 15/16 valid acc 14/16
Epoch 6 loss 0.768422630185047 train acc 16/16 valid acc 14/16
Epoch 6 loss 0.21647912493395555 train acc 16/16 valid acc 14/16
Epoch 6 loss 0.3450246416855649 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.008145948503345 train acc 14/16 valid acc 15/16
Epoch 6 loss 0.7690598967020614 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.626960993006454 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.5698678458948248 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.7099483547780822 train acc 16/16 valid acc 15/16
Epoch 6 loss 1.2215367779437056 train acc 15/16 valid acc 16/16
Epoch 6 loss 1.7181322834683863 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.2716984118979526 train acc 14/16 valid acc 14/16
Epoch 6 loss 1.2707640490704277 train acc 15/16 valid acc 15/16
Epoch 6 loss 0.7731097723493313 train acc 15/16 valid acc 14/16
Epoch 6 loss 0.9048595714169602 train acc 16/16 valid acc 14/16
Epoch 6 loss 1.053979635383252 train acc 13/16 valid acc 14/16
Epoch 6 loss 0.5181379184980952 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.3055674837854085 train acc 16/16 valid acc 14/16
Epoch 6 loss 0.6593896275148827 train acc 15/16 valid acc 13/16
Epoch 6 loss 0.6434353650476271 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.5013543559952166 train acc 14/16 valid acc 16/16
Epoch 6 loss 0.7106644595316032 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.31182581873765464 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.43286922721428506 train acc 15/16 valid acc 15/16
Epoch 6 loss 0.6128883187912163 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.6650081529438538 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.5797141132301296 train acc 14/16 valid acc 16/16
Epoch 6 loss 0.26631259594690554 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.5572444695172762 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.850039454721413 train acc 14/16 valid acc 16/16
Epoch 6 loss 0.731207765425277 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.5641202797963316 train acc 16/16 valid acc 16/16
Epoch 6 loss 1.122778335347492 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.2318545419834137 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.5024769053623415 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.6677271502262404 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.6195534419956685 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.7603838432757288 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.23328713780612545 train acc 16/16 valid acc 16/16
Epoch 6 loss 1.1365518238687757 train acc 14/16 valid acc 16/16
Epoch 6 loss 0.38879429440656454 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.6373997385980168 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.6801295104984186 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.4854953091504606 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.6881407393049865 train acc 15/16 valid acc 16/16
Epoch 6 loss 0.7011014747522745 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.6068742918646649 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.4838003777817895 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.6561858325399431 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.36564909936038537 train acc 16/16 valid acc 16/16
Epoch 6 loss 1.1485384686095361 train acc 15/16 valid acc 15/16
Epoch 6 loss 0.3904357344965129 train acc 15/16 valid acc 15/16
Epoch 6 loss 0.8808106357584189 train acc 14/16 valid acc 15/16
Epoch 6 loss 0.4607645800644872 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.1711853928453608 train acc 15/16 valid acc 15/16
Epoch 6 loss 0.5049196512396539 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.8291483634971923 train acc 15/16 valid acc 15/16
Epoch 6 loss 1.0184141698290057 train acc 16/16 valid acc 15/16
Epoch 6 loss 0.7548222564866987 train acc 14/16 valid acc 16/16
Epoch 6 loss 0.9006445312251388 train acc 16/16 valid acc 16/16
Epoch 6 loss 0.5745764255929866 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.01438940297961585 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.737871929763902 train acc 15/16 valid acc 16/16
Epoch 7 loss 1.6201394275309726 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.7115253377789165 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.5612792973824946 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.3597540617060132 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.8554001139293634 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.9540051242022671 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.579184297732336 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.35812474969498476 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.35666110665687106 train acc 15/16 valid acc 14/16
Epoch 7 loss 0.8570717411227223 train acc 15/16 valid acc 15/16
Epoch 7 loss 1.3203128979578485 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.32050201504261044 train acc 15/16 valid acc 14/16
Epoch 7 loss 1.2712970246205042 train acc 13/16 valid acc 15/16
Epoch 7 loss 0.6517083324014621 train acc 15/16 valid acc 14/16
Epoch 7 loss 0.49005683591785937 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.5770565968892891 train acc 16/16 valid acc 16/16
Epoch 7 loss 1.2751946715805889 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.3744201306883448 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.5972525837216428 train acc 15/16 valid acc 14/16
Epoch 7 loss 0.5696992727206807 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.25208030279715277 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.6770987081057906 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.4021012790783443 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.6976629968484249 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.30444298480610077 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.4671482430790347 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.3118390461040971 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.6882174787543116 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.5354849982264385 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.6037202087293461 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.6974206010102986 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.3049002935449615 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.4975952805945707 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.6054117121621753 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.5605776681716798 train acc 15/16 valid acc 16/16
Epoch 7 loss 1.0335927981378408 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.7724336881743294 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.2776874574827786 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.23619860803992188 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.6669115210191705 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.22358699472217283 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.24914950599505123 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.8330264686818081 train acc 14/16 valid acc 16/16
Epoch 7 loss 0.3986323074686492 train acc 15/16 valid acc 15/16
Epoch 7 loss 0.5225421836862001 train acc 15/16 valid acc 15/16
Epoch 7 loss 1.891131569095983 train acc 12/16 valid acc 16/16
Epoch 7 loss 0.9182666761424898 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.5217104071656206 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.3510213315404379 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.4191170305041617 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.45913715067795347 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.08167226883526238 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.6009684719575459 train acc 14/16 valid acc 16/16
Epoch 7 loss 0.8066091219972744 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.4388990525905016 train acc 15/16 valid acc 16/16
Epoch 7 loss 0.6139365502393578 train acc 16/16 valid acc 15/16
Epoch 7 loss 0.7287368591123833 train acc 15/16 valid acc 16/16
Epoch 7 loss 1.1155249957020348 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.18846329936562206 train acc 16/16 valid acc 16/16
Epoch 7 loss 0.2325782223479664 train acc 16/16 valid acc 15/16
Epoch 7 loss 1.0556814536904724 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.05226287641042134 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.5833991053837322 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.6820385441395795 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.6810356210590228 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.17630072386147258 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.40835977297637266 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.5234442448412939 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.5511731317062709 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.5185133992881978 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.24338092009104498 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.13539655664476483 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.42402435106840863 train acc 16/16 valid acc 15/16
Epoch 8 loss 1.3157715048924303 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.7555221169624066 train acc 14/16 valid acc 15/16
Epoch 8 loss 1.1646279976065341 train acc 14/16 valid acc 15/16
Epoch 8 loss 0.6745694798181031 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.6658940738920454 train acc 16/16 valid acc 16/16
Epoch 8 loss 1.3033334597821389 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.5327428136582936 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.6100911658957087 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.5134731560762233 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.24513125063050914 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.34647446834388296 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.1461670530035104 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.11966126478596177 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.5656461722294384 train acc 14/16 valid acc 16/16
Epoch 8 loss 0.8784562090558377 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.40366861916038826 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.33995238453594095 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.061431543816646915 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.714904228668022 train acc 15/16 valid acc 14/16
Epoch 8 loss 0.18625053390293822 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.5376410193131262 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.4003999858844561 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.9911238808069286 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.3088260038709364 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.336573180020011 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.8946218200776865 train acc 14/16 valid acc 14/16
Epoch 8 loss 0.5295527921203524 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.6376749181173369 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.31072866163305274 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.6651325171303425 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.4895459902587155 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.629281707250389 train acc 15/16 valid acc 14/16
Epoch 8 loss 0.5042230446668843 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.08344436515122974 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.3693919290136385 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.8097852117104041 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.7514035830912603 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.6254803658049225 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.5270890123390983 train acc 16/16 valid acc 16/16
Epoch 8 loss 1.5482488001048695 train acc 14/16 valid acc 16/16
Epoch 8 loss 0.38741220341296295 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.3351211073981565 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.3011030151459508 train acc 15/16 valid acc 15/16
Epoch 8 loss 0.25494390864453 train acc 16/16 valid acc 15/16
Epoch 8 loss 0.498887578796505 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.20832166027901583 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.9731784677133961 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.47307229040627224 train acc 16/16 valid acc 16/16
Epoch 8 loss 0.8202426442007037 train acc 15/16 valid acc 16/16
Epoch 8 loss 0.17253724476706747 train acc 16/16 valid acc 16/16
Epoch 8 loss 1.0368302809752425 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.0019976086676877715 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.2888031593294691 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.40552241960227536 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.5566970630859559 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.3490640685650659 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.13695418358607425 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.30283377538236794 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.39488259707424983 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.36577834488604155 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.30752172634653296 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.18241729056379216 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.4296761897356339 train acc 16/16 valid acc 16/16
Epoch 9 loss 1.100281914685402 train acc 15/16 valid acc 14/16
Epoch 9 loss 1.3629953279635325 train acc 14/16 valid acc 14/16
Epoch 9 loss 0.7592218574790882 train acc 15/16 valid acc 14/16
Epoch 9 loss 0.7632625322201703 train acc 15/16 valid acc 14/16
Epoch 9 loss 1.2037046456768752 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.6132983923774198 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.46359302696683985 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.37000085198754185 train acc 16/16 valid acc 15/16
Epoch 9 loss 1.3332573369727334 train acc 14/16 valid acc 16/16
Epoch 9 loss 0.2797432279170653 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.4158896760380667 train acc 14/16 valid acc 16/16
Epoch 9 loss 0.20419314863653065 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.3924268354844007 train acc 15/16 valid acc 16/16
Epoch 9 loss 1.289916706766773 train acc 15/16 valid acc 14/16
Epoch 9 loss 0.5712491005096259 train acc 16/16 valid acc 14/16
Epoch 9 loss 0.17630133801815503 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.3215862456913279 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.12378769084456054 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.28984666009552623 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.6321123930617264 train acc 14/16 valid acc 16/16
Epoch 9 loss 0.22037990893486603 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.11298475556660136 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.9431362369869054 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.17980235288087543 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.15774824392415432 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.28595361668647423 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.19135572132640677 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.051862571742278485 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.2350191018639657 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.19429999021032845 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.16250250310401645 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.23655381033052703 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.6220612158996411 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.2234407991099796 train acc 16/16 valid acc 16/16
Epoch 9 loss 0.34030757024529457 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.40711352515213484 train acc 15/16 valid acc 16/16
Epoch 9 loss 0.7613377805273489 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.8217646347851217 train acc 15/16 valid acc 15/16
Epoch 9 loss 1.0487520709737064 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.8315451323496201 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.972724839950089 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.3529629514689756 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.45930417267463475 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.3972464629887613 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.5663587083229863 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.4939592161249847 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.625369878696522 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.6559224424696729 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.17625630241926732 train acc 16/16 valid acc 15/16
Epoch 9 loss 0.48459374136401695 train acc 15/16 valid acc 15/16
Epoch 9 loss 0.44501730806852613 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.07907507054448643 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.17426203679283359 train acc 15/16 valid acc 15/16
Epoch 10 loss 1.2038284470714058 train acc 14/16 valid acc 14/16
Epoch 10 loss 0.7100773427204219 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.27685753570308125 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.8147042777250922 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.7997007794620774 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.865918984867848 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.44641336762613226 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.19356305356702083 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.9216179154720167 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.8930161637550174 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.612355889819126 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.25288347065400013 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.7230282829566352 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.8397059110853273 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.3972745478453693 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.7644048264902352 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.491081238587169 train acc 14/16 valid acc 15/16
Epoch 10 loss 0.3928150709961271 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.325852385089643 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.6771441274559773 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.15745862974357494 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.07726183230007906 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.13591970032940753 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.29436161604104555 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.2628440558963517 train acc 16/16 valid acc 14/16
Epoch 10 loss 0.9185330365826795 train acc 16/16 valid acc 14/16
Epoch 10 loss 0.5650954776667451 train acc 14/16 valid acc 14/16
Epoch 10 loss 0.1530040066280611 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.25928742467454247 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.17868994979622868 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.24957778101774047 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.8927949440733733 train acc 15/16 valid acc 15/16
Epoch 10 loss 0.9810373847514114 train acc 15/16 valid acc 14/16
Epoch 10 loss 0.6984109486811563 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.374435050896977 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.5098085433699737 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.38528224642197384 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.4743055509125789 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.45638737921227424 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.468209302669692 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.336793580741953 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.11309388896201009 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.5798275642619353 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.17449047131696357 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.7255157574540108 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.9806204175654403 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.10398981888247093 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.20132726814275959 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.4076519235019063 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.3956370169746519 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.6324903161525283 train acc 16/16 valid acc 16/16
Epoch 10 loss 0.8403334716758581 train acc 14/16 valid acc 16/16
Epoch 10 loss 0.4713650813183664 train acc 15/16 valid acc 13/16
Epoch 10 loss 0.5757175785953477 train acc 15/16 valid acc 13/16
Epoch 10 loss 0.44694987825599786 train acc 16/16 valid acc 14/16
Epoch 10 loss 0.5869308095308399 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.674790977830289 train acc 15/16 valid acc 16/16
Epoch 10 loss 1.4226160420321146 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.8487036904561245 train acc 16/16 valid acc 15/16
Epoch 10 loss 0.8499983296925169 train acc 15/16 valid acc 16/16
Epoch 10 loss 0.6690831117301143 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.039625321804357246 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.47036648742695203 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.195233306405925 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.55684743838452 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.18421818529289283 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.21237253929975008 train acc 15/16 valid acc 16/16
Epoch 11 loss 1.3169388879669786 train acc 15/16 valid acc 16/16
Epoch 11 loss 1.4261204227993276 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.5704942402226324 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.04895558792566466 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.9000897321744569 train acc 15/16 valid acc 16/16
Epoch 11 loss 1.3459604003985493 train acc 15/16 valid acc 16/16
Epoch 11 loss 1.1807979690285444 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.546397051339883 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.7729589909682728 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.901819717277395 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.4759420767135673 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.5998161988198264 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.1875572643880381 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.1102699456080861 train acc 15/16 valid acc 16/16
Epoch 11 loss 1.0059480125368065 train acc 13/16 valid acc 15/16
Epoch 11 loss 0.3200358451607342 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.06544770736394682 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.06294064336006004 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.12972438094401695 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.7302602851033322 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.8460973578378543 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.2774592500864466 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.8487707237575514 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.07505716256185513 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.43152737909701805 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.334602602215914 train acc 14/16 valid acc 16/16
Epoch 11 loss 0.3619639283108425 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.6250803570504957 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.6886492854656852 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.444750997932884 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.18164280401729943 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.3056046264359465 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.39276076887777644 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.0955822827953458 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.17401660270399133 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.3027906539259091 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.3210276482142656 train acc 14/16 valid acc 15/16
Epoch 11 loss 0.10946462938448338 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.5476488164612923 train acc 14/16 valid acc 15/16
Epoch 11 loss 0.06234225653147064 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.9677268983690326 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.729981946037234 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.9571767257484858 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.5681548324290302 train acc 15/16 valid acc 16/16
Epoch 11 loss 0.5693237491804181 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.36495574819904597 train acc 16/16 valid acc 14/16
Epoch 11 loss 0.10959946125152545 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.09160296656979412 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.31034910840042335 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.721618182672326 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.317534761954595 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.2803081870157196 train acc 16/16 valid acc 15/16
Epoch 11 loss 0.5333227580048815 train acc 15/16 valid acc 14/16
Epoch 11 loss 0.91262159723122 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.6148903541562809 train acc 15/16 valid acc 15/16
Epoch 11 loss 0.16419764007578683 train acc 16/16 valid acc 16/16
Epoch 11 loss 0.37274429441117324 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.0030973834162446616 train acc 16/16 valid acc 14/16
Epoch 12 loss 0.5819190140739208 train acc 15/16 valid acc 16/16
Epoch 12 loss 1.2661635831145874 train acc 14/16 valid acc 16/16
Epoch 12 loss 0.4971924566515147 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.08256297277409248 train acc 16/16 valid acc 14/16
Epoch 12 loss 0.454645221908202 train acc 15/16 valid acc 14/16
Epoch 12 loss 0.666274094463204 train acc 13/16 valid acc 15/16
Epoch 12 loss 0.6062765186289736 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.23982433343101892 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.4144141724733177 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.23156205438837665 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.5641589118198523 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.7640737327977687 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.6703021357397676 train acc 15/16 valid acc 14/16
Epoch 12 loss 0.23293542737017878 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.6428798976112955 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.881476302929731 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.2667629437300217 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.36023449269149105 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.20357510151294322 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.7634484477636236 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.160431214396644 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.29145690811816044 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.7287328826683642 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.18071394158942805 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.7021776495798507 train acc 15/16 valid acc 14/16
Epoch 12 loss 0.4902855031549279 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.1482811561881519 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.04838925956067118 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.04306346580507919 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.4707527349009485 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.3823651949601242 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.11218613994239073 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.41576130470124417 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.6086404883982288 train acc 14/16 valid acc 16/16
Epoch 12 loss 0.44603912403376755 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.20850836206863343 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.22363058280086529 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.11343504381853456 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.2378248474409339 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.20087703165118884 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.8155806386431871 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.07856272095227831 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.10996280591360937 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.38626907912088443 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.3161497789112898 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.3189252750943558 train acc 15/16 valid acc 15/16
Epoch 12 loss 1.109667587186996 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.16962592616147826 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.46921577796608677 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.2398740438987846 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.25270103132928634 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.4295699492089382 train acc 15/16 valid acc 16/16
Epoch 12 loss 0.49808775284090856 train acc 16/16 valid acc 16/16
Epoch 12 loss 0.33551482486682693 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.26931886039038494 train acc 15/16 valid acc 15/16
Epoch 12 loss 0.36497877990181427 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.1876487935799865 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.5862722311909565 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.4309942378667852 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.10870277734150366 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.1693770565926782 train acc 16/16 valid acc 15/16
Epoch 12 loss 0.5155720662831574 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.012987502433157204 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.8395447892413479 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.3329774506796051 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.267276824315707 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.10115549363741587 train acc 16/16 valid acc 14/16
Epoch 13 loss 0.19047602209005143 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.9162827058270981 train acc 15/16 valid acc 14/16
Epoch 13 loss 0.3298449076972966 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.4434026642835139 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.18891102425934175 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.2605756312911992 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.22833335187937487 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.49345360796972926 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.6493890126902351 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.2867276232278039 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.858474012437097 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.6389676166817454 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.9005256784855891 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.5644080245263825 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.07235529161719603 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.5322238452226871 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.1703008905498109 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.1463717885161286 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.13624487908285937 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.06041426294273627 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.1349968121872071 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.18797565288762233 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.03020079800278882 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.18377276999219033 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.15745461728699844 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.48310923415994367 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.2212619567586173 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.5554454256149005 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.34638644371960065 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.4001775331162069 train acc 14/16 valid acc 16/16
Epoch 13 loss 0.6678386553635023 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.34051101902099024 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.3044160646316131 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.2558503913180158 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.2490671302073799 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.2780900822520796 train acc 16/16 valid acc 14/16
Epoch 13 loss 0.1456132774187718 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.016481153920115434 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.2396740946591359 train acc 16/16 valid acc 14/16
Epoch 13 loss 0.6196225988235146 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.41631086199756107 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.22153494040545774 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.5700062639973656 train acc 15/16 valid acc 16/16
Epoch 13 loss 0.5366121884726264 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.2622724378627038 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.12493571001836543 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.14120374840079197 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.08957750748565235 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.13255638613639195 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.08625773848408776 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.13626622084980672 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.3510837760993671 train acc 15/16 valid acc 15/16
Epoch 13 loss 0.16409354189063635 train acc 15/16 valid acc 14/16
Epoch 13 loss 0.8184521426878507 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.5658496557225465 train acc 16/16 valid acc 16/16
Epoch 13 loss 0.8149399929031891 train acc 13/16 valid acc 16/16
Epoch 13 loss 0.14617970626785967 train acc 16/16 valid acc 15/16
Epoch 13 loss 0.4057697116135386 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.002314806352539601 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.5836057319913357 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.5419962600529764 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.3355967017683261 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.19832249095189533 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.19814857124969554 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.7745021892158064 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.24579965336188778 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.6322721913649804 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.33242625344092674 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.1483717616584031 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.6273994561680676 train acc 15/16 valid acc 14/16
Epoch 14 loss 0.8783787368680637 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.7080711955447776 train acc 14/16 valid acc 14/16
Epoch 14 loss 0.9200734791510605 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.7416883321076277 train acc 15/16 valid acc 14/16
Epoch 14 loss 0.4679955549888997 train acc 16/16 valid acc 16/16
Epoch 14 loss 1.2350038756862145 train acc 13/16 valid acc 15/16
Epoch 14 loss 0.8507237904288821 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.2955948007116626 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.2787201623992976 train acc 15/16 valid acc 13/16
Epoch 14 loss 0.339551047476328 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.04305622129751466 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.2360171121622594 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.5323240819170808 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.5927596606933292 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.3460012180248817 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.11053268676460826 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.39078581012665264 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.18221273507608507 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.2890132012407039 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.17430515426165843 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.30395950504879027 train acc 16/16 valid acc 14/16
Epoch 14 loss 0.23553786171182306 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.4388348879504577 train acc 14/16 valid acc 15/16
Epoch 14 loss 0.3439573227823269 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.20097569313983854 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.2875844592342202 train acc 16/16 valid acc 15/16
Epoch 14 loss 1.3390133882284976 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.809204047366003 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.11357292617057832 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.23549939675324233 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.4998788845058738 train acc 14/16 valid acc 15/16
Epoch 14 loss 0.2595856564020032 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.4957343622989858 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.12136253728140461 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.32059409569444697 train acc 15/16 valid acc 15/16
Epoch 14 loss 1.255164193804094 train acc 15/16 valid acc 14/16
Epoch 14 loss 0.4149965728227086 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.25594828635384226 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.11681557602210639 train acc 16/16 valid acc 16/16
Epoch 14 loss 0.17384013801481757 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.6229286058447981 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.05027333553377349 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.3813574998991258 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.6921515713328357 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.5428170189235327 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.1704537009138586 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.23040395627897994 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.4072955452226867 train acc 16/16 valid acc 15/16
Epoch 14 loss 0.45463592801183916 train acc 15/16 valid acc 15/16
Epoch 14 loss 0.3758138883432236 train acc 15/16 valid acc 16/16
Epoch 14 loss 0.6517800506130944 train acc 15/16 valid acc 16/16
Epoch 15 loss 0.0006804294668897608 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.413084500362942 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.8066298399133159 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.27168949687628646 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.008075561972835349 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.24832374137228047 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.33546950297230854 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.4095767021768479 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.11940186940585214 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.19966922824091776 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.15700235206901683 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.3921637793818147 train acc 16/16 valid acc 16/16
Epoch 15 loss 1.1296858873258677 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.10500398469573173 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.3750478806980016 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.41665783832440384 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.4145620897927365 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.7495659225628224 train acc 14/16 valid acc 15/16
Epoch 15 loss 0.10728275595170116 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.09782457254629412 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.18176324576904646 train acc 15/16 valid acc 16/16
Epoch 15 loss 0.39350330884912693 train acc 15/16 valid acc 16/16
Epoch 15 loss 0.40596913607053137 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.3616111388636657 train acc 15/16 valid acc 16/16
Epoch 15 loss 0.24667480173564882 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.6347804997731017 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.5694781646776889 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.20530902055740036 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.3852625931601265 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.10395242162671496 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.03803942176172199 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.07214226816168859 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.22094297102805455 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.3678885253579486 train acc 16/16 valid acc 14/16
Epoch 15 loss 0.22225770147058277 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.11461558988809571 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.03742777789272725 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.3022431093410201 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.6776610099279443 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.10032074383258258 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.011204965103037394 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.06343613639452421 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.08299510564598274 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.6649811338274625 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.29551251021771996 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.1214344436088351 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.2504422020153939 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.3604693520343004 train acc 16/16 valid acc 16/16
Epoch 15 loss 0.44214213180633505 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.44877317064303895 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.5561340813079649 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.7083935089214413 train acc 16/16 valid acc 14/16
Epoch 15 loss 0.3764952192763057 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.2750905006386679 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.13180782274778852 train acc 16/16 valid acc 15/16
Epoch 15 loss 0.8007716860584129 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.7248001605772931 train acc 15/16 valid acc 15/16
Epoch 15 loss 0.1285375984929232 train acc 16/16 valid acc 14/16
Epoch 15 loss 0.9081913717373292 train acc 16/16 valid acc 14/16
Epoch 15 loss 1.0743694517425793 train acc 14/16 valid acc 15/16
Epoch 15 loss 0.2122903170414323 train acc 16/16 valid acc 14/16
Epoch 15 loss 0.17126409200677775 train acc 16/16 valid acc 14/16
Epoch 15 loss 0.10896912894141293 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.006402586648772092 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.7209464401982888 train acc 14/16 valid acc 14/16
Epoch 16 loss 0.3810546306238478 train acc 15/16 valid acc 14/16
Epoch 16 loss 0.5577116597716621 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.11427376111256887 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.11684733698028278 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.3033710971402533 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.31515144434668624 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.397757593293769 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.24089532188672444 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.2139540112197492 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.621023120827844 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.4269879101522645 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.09825428012331111 train acc 15/16 valid acc 14/16
Epoch 16 loss 0.21378605487637378 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.8081544473124909 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.4773338964500374 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.6485600902476507 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.174505280859928 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.2268990129109968 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.6113598703750025 train acc 15/16 valid acc 14/16
Epoch 16 loss 0.3750199032103964 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.2222693087854505 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.02799604711604872 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.0618114226520812 train acc 15/16 valid acc 15/16
Epoch 16 loss 2.6594446710259727 train acc 16/16 valid acc 14/16
Epoch 16 loss 0.34660077077036283 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.4894837458426866 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.31461536396689826 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.23610759934175896 train acc 16/16 valid acc 15/16
Epoch 16 loss 1.1502652429326679 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.7583779598484787 train acc 14/16 valid acc 13/16
Epoch 16 loss 0.558953708933961 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.6893274472085701 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.4591544528579974 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.2166175046100472 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.1388090161546343 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.01872559275641091 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.14925826766563702 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.18480387676027366 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.09013126553241507 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.5376930666975672 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.17519639926989547 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.16634434614613924 train acc 16/16 valid acc 15/16
Epoch 16 loss 1.026917873412736 train acc 14/16 valid acc 16/16
Epoch 16 loss 0.10737041407341835 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.4378396056553385 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.47367834885045257 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.06198119248356969 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.31451643656607065 train acc 15/16 valid acc 15/16
Epoch 16 loss 0.12213117947636344 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.18697123590971412 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.3451033873885564 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.024542358995149297 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.22843778227806666 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.32658224636733046 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.3925391785927619 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.08181797263525033 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.4665747403459516 train acc 15/16 valid acc 16/16
Epoch 16 loss 0.544767605276429 train acc 16/16 valid acc 16/16
Epoch 16 loss 0.3340523343686591 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.09767419652193927 train acc 16/16 valid acc 15/16
Epoch 16 loss 0.2715098919724753 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.0760093773929754 train acc 15/16 valid acc 15/16
Epoch 17 loss 0.06192234737428977 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.5339546509088933 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.27633260465383824 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.17501043435457636 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.2985998148852196 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.6914319298899826 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.6529566130318692 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.4706186365476293 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.2998491151373933 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.5166989267785208 train acc 14/16 valid acc 16/16
Epoch 17 loss 1.1349922173792284 train acc 15/16 valid acc 15/16
Epoch 17 loss 0.6025902303432893 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.6005839069895939 train acc 14/16 valid acc 16/16
Epoch 17 loss 0.8043882663220634 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.42030882496610666 train acc 15/16 valid acc 15/16
Epoch 17 loss 0.6779989755947219 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.2968573963021158 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.07431119973749467 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.16023299324378248 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.14341224477288572 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.23249007436304442 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.2945972540437784 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.05700338446848686 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.5404423652276822 train acc 15/16 valid acc 14/16
Epoch 17 loss 0.6902518698426727 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.3989379891983112 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.3155338999173153 train acc 16/16 valid acc 14/16
Epoch 17 loss 0.16330022127733876 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.5299132266703619 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.25422851512661115 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.05598116026377283 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.28565208268464004 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.315222363408421 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.1389040236922 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.46784300522555844 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.47074168012846396 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.09196741847682938 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.1734316922837695 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.09789759231958373 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.4352156095878042 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.5283979519152033 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.11406048735784409 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.08968529423270147 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.42586507237084803 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.015938684926068196 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.3880060478766412 train acc 15/16 valid acc 16/16
Epoch 17 loss 0.584664724835803 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.2222064501052201 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.2888034806302128 train acc 16/16 valid acc 16/16
Epoch 17 loss 1.191245413990171 train acc 15/16 valid acc 15/16
Epoch 17 loss 0.5906857537651123 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.09653318394045006 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.17204159132784472 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.08420095842546166 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.21603973214871036 train acc 15/16 valid acc 15/16
Epoch 17 loss 0.3777140882595333 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.2081788929166593 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.2057291630056484 train acc 16/16 valid acc 16/16
Epoch 17 loss 0.24935182435755082 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.05006326098621582 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.03787338447718 train acc 16/16 valid acc 15/16
Epoch 17 loss 0.1681510677086838 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.0035995450728619326 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.27293357937296014 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.2567872671683999 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.4095852905218763 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.28733335513002906 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.3127742768182756 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.5192002576311957 train acc 15/16 valid acc 16/16
Epoch 18 loss 0.28993683356247424 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.7341507296719718 train acc 15/16 valid acc 16/16
Epoch 18 loss 0.008694296644052874 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.25366505650245574 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.1814908570042616 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.8336075427649439 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.2971478429731838 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.36036122714578367 train acc 15/16 valid acc 16/16
Epoch 18 loss 0.5570800170727694 train acc 15/16 valid acc 14/16
Epoch 18 loss 0.6686406069905837 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.7217936459957762 train acc 15/16 valid acc 14/16
Epoch 18 loss 0.5382385351499701 train acc 15/16 valid acc 16/16
Epoch 18 loss 0.4201157308220802 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.3877128739134316 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.18520317096967276 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.10977188946916634 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.174499860256668 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.22633005822980715 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.9188448765942855 train acc 14/16 valid acc 16/16
Epoch 18 loss 0.44180258922418086 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.05168048897076533 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.24998315465190926 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.25356802269174533 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.15818477660611838 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.26346709478840924 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.1536244125763523 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.08820377095458817 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.2087943104627019 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.6054944616520879 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.041773881527170986 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.17960197206462616 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.3351508964009675 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.10669328574600734 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.01598619654311923 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.20923112888082993 train acc 15/16 valid acc 16/16
Epoch 18 loss 0.39455444521611077 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.03806269522049772 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.1598175134840518 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.0838280619040721 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.08322005039677371 train acc 15/16 valid acc 15/16
Epoch 18 loss 0.1947601595110009 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.00887490468425935 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.21268979024440013 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.1056429086025758 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.38902828392891764 train acc 16/16 valid acc 16/16
Epoch 18 loss 0.7452989047510092 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.04029351837209561 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.1884701350905047 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.5897380725704542 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.3562849536773022 train acc 15/16 valid acc 14/16
Epoch 18 loss 0.19031690153719558 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.5329065942893729 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.16539573677448768 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.023877893590836946 train acc 16/16 valid acc 14/16
Epoch 18 loss 0.039910575820805844 train acc 16/16 valid acc 15/16
Epoch 18 loss 0.9337896916403174 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.008274253821332497 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.26763719274457953 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.45348500962557525 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.4712332122811873 train acc 15/16 valid acc 14/16
Epoch 19 loss 0.07340349263999069 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.009336502257412008 train acc 16/16 valid acc 14/16
Epoch 19 loss 0.4951263557609913 train acc 15/16 valid acc 14/16
Epoch 19 loss 0.49147224306930165 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.15416784094726166 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.07169123949396215 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.05345672220236665 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.07563740383442358 train acc 16/16 valid acc 15/16
Epoch 19 loss 1.0090687466517134 train acc 15/16 valid acc 14/16
Epoch 19 loss 0.22961717297965337 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.2892494612726215 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.8678119968136669 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.468887356002712 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.3681003019481379 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.4321616319387073 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.035394013547189795 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.3302165603471069 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.23001412207364536 train acc 15/16 valid acc 16/16
Epoch 19 loss 0.6690411608851636 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.1270257768709831 train acc 16/16 valid acc 14/16
Epoch 19 loss 0.023233698676498958 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.17786528708387242 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.04417116121665821 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.06771626149068732 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.3601285608690215 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.04813848958461555 train acc 16/16 valid acc 16/16
Epoch 19 loss 1.022914264722358 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.4410989787623758 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.07684893640840387 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.14653849416086495 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.42484996378214057 train acc 14/16 valid acc 15/16
Epoch 19 loss 0.4591821883692281 train acc 14/16 valid acc 15/16
Epoch 19 loss 0.6898780254662413 train acc 15/16 valid acc 16/16
Epoch 19 loss 0.5142110585039639 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.23064603384254279 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.09683165686916925 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.380318934186047 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.36786622093237414 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.2902478167917527 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.18548935441269476 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.25967509670391137 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.13636452929057588 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.16220087673792444 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.11468763992091348 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.0320240063827565 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.17009399662932081 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.4439729235679292 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.3155478860890078 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.15153838576960318 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.13460220663212616 train acc 16/16 valid acc 16/16
Epoch 19 loss 0.3597862084328942 train acc 15/16 valid acc 16/16
Epoch 19 loss 0.27542776180530654 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.730921461313245 train acc 15/16 valid acc 16/16
Epoch 19 loss 0.1945712394068517 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.7505304610211162 train acc 15/16 valid acc 15/16
Epoch 19 loss 0.3380991352100173 train acc 16/16 valid acc 13/16
Epoch 19 loss 0.793730173199732 train acc 16/16 valid acc 14/16
Epoch 19 loss 0.3248826883806251 train acc 16/16 valid acc 15/16
Epoch 19 loss 0.3069130627124561 train acc 15/16 valid acc 16/16
Epoch 20 loss 0.0008217698907885097 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.7478290879818006 train acc 15/16 valid acc 16/16
Epoch 20 loss 0.10387475224060938 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.3584293116426601 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.3285010953118067 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.06573531411829829 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.11073348390135562 train acc 15/16 valid acc 16/16
Epoch 20 loss 0.3619016352619322 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.2843132100441585 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.14613093328175186 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.24115415307171095 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.09178746134606025 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.4990532326638756 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.31023290191802666 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.2673736076349309 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.517592702943089 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.4366274769761158 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.3083662773060818 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.24038923577599158 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.3910298917031662 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.10081522095956938 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.16276378130325644 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.0656204379139806 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.4679956355264656 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.04937532449779101 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.2384087931039498 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.02491059632726851 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.568367624270075 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.02188165526606666 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.09123441288228193 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.38340987974315943 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.2569289202262729 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.46385489460317 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.6049454383443327 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.19886555978424073 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.2067782593997841 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.01983352193200244 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.1780758056355252 train acc 16/16 valid acc 16/16
Epoch 20 loss 0.5087201876297113 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.3410001860843242 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.061967235814794575 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.7836511543006043 train acc 15/16 valid acc 15/16
Epoch 20 loss 0.2922261904306076 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.21167358175930098 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.20367684766680016 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.34804777629145506 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.4240516651484425 train acc 15/16 valid acc 14/16
Epoch 20 loss 1.0174528766381101 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.19396997004151645 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.2797232645715824 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.055204917573697505 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.6876313997642196 train acc 16/16 valid acc 13/16
Epoch 20 loss 0.6278306870263002 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.07206442472388724 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.26506166111722085 train acc 15/16 valid acc 14/16
Epoch 20 loss 0.2071817101489749 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.5298562479958305 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.17905349474136023 train acc 16/16 valid acc 15/16
Epoch 20 loss 0.11152544015139605 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.26635568445069524 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.34131531595367004 train acc 16/16 valid acc 13/16
Epoch 20 loss 0.26594039707526046 train acc 16/16 valid acc 14/16
Epoch 20 loss 0.7676424588499046 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.016777082004064185 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.06933560503591345 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.13937853979119302 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.42088903475991185 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.02506282413784367 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.5182467208083493 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.2658444596282107 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.24596239280607127 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.5685220526315263 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.7595893339431679 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.2578511325420027 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.6320353810448565 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.3375725771054259 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.4695192417627563 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.23663055839837566 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.38289898957180896 train acc 14/16 valid acc 15/16
Epoch 21 loss 0.3951050581967837 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.8501493340347333 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.27868772164959144 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.04940152156578476 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.6231163144172509 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.13856221201386715 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.11107876554654246 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.32960442404300433 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.10896157552169833 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.3518630677739255 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.3842261078331146 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.11451405684017252 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.15183643291417595 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.22400535209690522 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.033647105818782094 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.06825802200828544 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.3639070711934083 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.15396408570847947 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.17461981251373782 train acc 14/16 valid acc 14/16
Epoch 21 loss 0.04921791114111064 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.06412427174622602 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.0557080114179144 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.15459197960380056 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.5118815604114046 train acc 14/16 valid acc 13/16
Epoch 21 loss 0.33667514590187586 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.23183897315750226 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.07322206449042358 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.026272525471917236 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.7206771926301337 train acc 15/16 valid acc 14/16
Epoch 21 loss 0.040078411169242545 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.11252292751144 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.7924446769934006 train acc 15/16 valid acc 14/16
Epoch 21 loss 0.46528465621037884 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.05143307366005163 train acc 16/16 valid acc 14/16
Epoch 21 loss 0.16647754440319315 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.12417870382910316 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.07201385617654397 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.18518685977083843 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.08057526477451173 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.11182268594642983 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.1778581997779234 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.5799992220133734 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.37914015451322736 train acc 15/16 valid acc 15/16
Epoch 21 loss 0.4475988683903844 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.002748905414157079 train acc 16/16 valid acc 15/16
Epoch 21 loss 0.053437136437320946 train acc 16/16 valid acc 16/16
Epoch 21 loss 0.08467701095411863 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.002191518667663431 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.1925842071457295 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.256221678705227 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.7185680607923041 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.1401888565839208 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.20711559511044292 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.7247763161958538 train acc 15/16 valid acc 15/16
Epoch 22 loss 0.2412118902231088 train acc 15/16 valid acc 15/16
Epoch 22 loss 0.08017424995491094 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.17029528943606664 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.15649193060967806 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.4837713337770135 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.12697697022922808 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.8077848474147868 train acc 15/16 valid acc 16/16
Epoch 22 loss 0.7232230773645645 train acc 15/16 valid acc 15/16
Epoch 22 loss 0.424667676020473 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.3621839669861705 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.4579375995401951 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.3241009206315066 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.21042827628276956 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.5634132225049765 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.41711365315344373 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.06165147642233948 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.2604165206753456 train acc 14/16 valid acc 14/16
Epoch 22 loss 0.22265876700859807 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.5102918375560179 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.09857725961376582 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.06706236003557385 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.24041825439981362 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.011841386505069787 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.29311265361524386 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.12352696163760327 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.08155423749603642 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.14220696500231558 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.34441868620302896 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.23928389634374328 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.3525257168787316 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.2021962297306684 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.07314619835256338 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.3571112047473127 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.28711769126029557 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.1275624827599206 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.0983907639552053 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.03564545382759543 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.11023797815221384 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.3468405971030384 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.20803582268125864 train acc 15/16 valid acc 14/16
Epoch 22 loss 0.7246751803637761 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.040280281464310744 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.4101053456982026 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.47473341615755477 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.2548471573927782 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.08023654513312284 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.0792746512293444 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.21535246391864804 train acc 15/16 valid acc 16/16
Epoch 22 loss 0.20092288023232313 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.23946584013883015 train acc 16/16 valid acc 16/16
Epoch 22 loss 0.030395382371499934 train acc 16/16 valid acc 16/16
Epoch 22 loss 1.1641612122120866 train acc 14/16 valid acc 16/16
Epoch 22 loss 0.22911611646952532 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.2246292267647745 train acc 16/16 valid acc 14/16
Epoch 22 loss 0.10112301433932773 train acc 16/16 valid acc 15/16
Epoch 22 loss 0.0431748463741067 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.025885436704970892 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.09247519973908147 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.28825665505739895 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.050665458715915455 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.059833761511574365 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.1198240310549355 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.46814445925068093 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.9444233019051311 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.46633657776942583 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.16772645903634245 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.08664120668361215 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.2505434169845054 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.3728714066837334 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.23022017753463025 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.24018692882236556 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.2158501604086372 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.8394611802790309 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.43667256207366273 train acc 14/16 valid acc 15/16
Epoch 23 loss 0.4492061775637125 train acc 15/16 valid acc 16/16
Epoch 23 loss 0.376387773454187 train acc 15/16 valid acc 16/16
Epoch 23 loss 0.18217637131526415 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.0353277152161097 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.22788001591536042 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.2098213478528791 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.18770334890022644 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.3945820711354805 train acc 15/16 valid acc 16/16
Epoch 23 loss 0.8189689221265473 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.48889198833331016 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.7345042657031395 train acc 15/16 valid acc 14/16
Epoch 23 loss 0.18835723793184922 train acc 15/16 valid acc 14/16
Epoch 23 loss 0.11260045648650253 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.6450400541738959 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.05560226789582743 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.08283650691586134 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.24291267622366136 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.15867182277365954 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.12738257539878328 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.0774785442793009 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.11844406492200109 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.09304068736272475 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.012869047554620299 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.17744986628630377 train acc 16/16 valid acc 16/16
Epoch 23 loss 0.13223999306264603 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.16947810074591851 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.6335348158862297 train acc 14/16 valid acc 15/16
Epoch 23 loss 0.08339899511950127 train acc 15/16 valid acc 14/16
Epoch 23 loss 0.26955132680244426 train acc 15/16 valid acc 15/16
Epoch 23 loss 0.19811679426468598 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.12493833338357149 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.13609650558477443 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.49721757793186866 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.5389073369933493 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.19000316946020124 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.0721742721312586 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.35634512476352853 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.05293912796484831 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.22131691386294858 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.2686726369054962 train acc 15/16 valid acc 14/16
Epoch 23 loss 0.19325931725893095 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.7690197642305363 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.01492983257698092 train acc 16/16 valid acc 15/16
Epoch 23 loss 0.1835898202433261 train acc 16/16 valid acc 14/16
Epoch 23 loss 0.21607126027573287 train acc 15/16 valid acc 14/16
Epoch 24 loss 0.001179670070607801 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.2689719473120884 train acc 15/16 valid acc 14/16
Epoch 24 loss 0.09094733286121964 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.20739023704359044 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.04331233251283539 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.6279292641903681 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.2222167234339584 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.3085909316736266 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.014995732692653623 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.03764313292250794 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.18299253989878472 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.28489691636375314 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.05435670840103341 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.2021231632383947 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.10651011301180463 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.17960500523058487 train acc 15/16 valid acc 14/16
Epoch 24 loss 0.336840398071264 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.0866258941381397 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.12345301507515849 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.05030567050503483 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.06381054353553381 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.10005117998208449 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.0639670044851271 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.033642209623613986 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.07996916763871623 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.018616205023948992 train acc 16/16 valid acc 14/16
Epoch 24 loss 0.06988961914158236 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.15027534193817343 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.010687157930906245 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.04756526019353398 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.5155234420025907 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.0280070336809349 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.056344294371482195 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.23360931710342142 train acc 16/16 valid acc 15/16
Epoch 24 loss 1.1521151398528857 train acc 15/16 valid acc 16/16
Epoch 24 loss 0.10226119720409899 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.1427160029897276 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.18699426335258357 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.47138263131716557 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.1463238912176626 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.4113932609103142 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.6487561529697703 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.08983042377363695 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.4958714240647475 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.2897659537646218 train acc 15/16 valid acc 16/16
Epoch 24 loss 0.050161747847254745 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.19728480996974113 train acc 15/16 valid acc 16/16
Epoch 24 loss 0.6349608388199619 train acc 15/16 valid acc 16/16
Epoch 24 loss 0.04269237555858992 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.0788468227120715 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.08180578999994195 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.045940151039512614 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.18264387358905232 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.07003969805160404 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.20231793744285198 train acc 15/16 valid acc 16/16
Epoch 24 loss 0.22128018234707678 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.09570702236427014 train acc 16/16 valid acc 16/16
Epoch 24 loss 0.3357487183064868 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.03757608288409979 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.20465598988657902 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.051288486187187204 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.16952067047369854 train acc 16/16 valid acc 15/16
Epoch 24 loss 0.16675610461147986 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.030329159731355363 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.16948039970096143 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.2798881433862338 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.09903369363664127 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.1652285945197548 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.21279928355259153 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.19945072969651712 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.2546888772199272 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.2230562543874392 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.02599125995879503 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.04558891653764394 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.4511885221348225 train acc 15/16 valid acc 15/16
Epoch 25 loss 0.5964285987192898 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.4598480474964106 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.18088938062520266 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.268901641906648 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.25015748032313995 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.20584725424768568 train acc 15/16 valid acc 15/16
Epoch 25 loss 0.46917612464032454 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.02912512373217976 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.07291314537949838 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.19582591315785525 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.0069619309329869 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.1363523639771677 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.00389472356052165 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.037455004665877575 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.31489846123297227 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.09501101396979587 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.08923222089538264 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.05983293694911689 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.10163506259682824 train acc 15/16 valid acc 15/16
Epoch 25 loss 0.06862859791668104 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.14835857831912233 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.518848569589516 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.40887563588543147 train acc 14/16 valid acc 16/16
Epoch 25 loss 0.8434491973350965 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.21092179535719796 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.04406935921321202 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.02602467140690751 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.2821587042049301 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.08563933684779063 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.13035663460167163 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.03387423476620699 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.016744494232837522 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.5811077034586244 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.06527812303212951 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.14342939531458362 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.13813618588506524 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.46153839570049926 train acc 16/16 valid acc 16/16
Epoch 25 loss 1.1084544192885155 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.4078808590779884 train acc 15/16 valid acc 16/16
Epoch 25 loss 0.1781257978132036 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.5601831310157315 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.1014184287900227 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.17591017788059743 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.42930310710758907 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.24254638605636544 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.23909086976759353 train acc 16/16 valid acc 16/16
Epoch 25 loss 0.45808988905359926 train acc 15/16 valid acc 14/16
Epoch 25 loss 0.826262068282134 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.016138476145918405 train acc 16/16 valid acc 14/16
Epoch 25 loss 0.2969625230535249 train acc 16/16 valid acc 15/16
Epoch 25 loss 0.170862446623597 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.0038977503263485074 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.7219461574388356 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.15493941929500676 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.08869940157504339 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.3264564305137275 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.31354117119041375 train acc 15/16 valid acc 14/16
Epoch 26 loss 0.10286202161482824 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.3718826958677138 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.3551726110622901 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.007571763429197577 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.3643815653635859 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.24801330065786958 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.13798434737260687 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.1349406598519805 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.23438851615470183 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.40954726847502293 train acc 15/16 valid acc 14/16
Epoch 26 loss 0.20147070296631026 train acc 15/16 valid acc 14/16
Epoch 26 loss 0.17436067884725281 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.5231255702267028 train acc 15/16 valid acc 14/16
Epoch 26 loss 0.21075615004346182 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.2608262525400936 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.01923520202475389 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.13950714550437837 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.09062470585402577 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.049695741098597555 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.20902409028858 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.31316950908308017 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.3376791543316001 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.553109916571448 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.03298233172163284 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.13103292700836877 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.07226552346616985 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.03738670097170502 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.08067217407006379 train acc 16/16 valid acc 14/16
Epoch 26 loss 0.14205250111993095 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.2180235663523175 train acc 16/16 valid acc 16/16
Epoch 26 loss 0.07570133255337617 train acc 16/16 valid acc 16/16
Epoch 26 loss 0.5168924772051495 train acc 16/16 valid acc 16/16
Epoch 26 loss 0.408036124691049 train acc 16/16 valid acc 16/16
Epoch 26 loss 0.11976741032975448 train acc 16/16 valid acc 16/16
Epoch 26 loss 0.05389696180266584 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.12704995220981097 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.10315266479557006 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.33862334820830126 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.3253930327768555 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.006164349297072691 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.12285880402454039 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.4344705362291049 train acc 15/16 valid acc 15/16
Epoch 26 loss 0.3481796493176629 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.4047408364170191 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.031443070626849146 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.13616411853956337 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.29461444656231744 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.02494049180662382 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.4857402275109366 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.09572946117974022 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.21178519613571545 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.25751601357157755 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.17885168180358668 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.40487860376462625 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.027701865038248393 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.18223465527540023 train acc 16/16 valid acc 15/16
Epoch 26 loss 0.14500658515106904 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.0008134095547758237 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.06417375175297649 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.6024141015677777 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.563862644366266 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.18541914166416498 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.1239654496905253 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.07996541067557231 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.13888707427396854 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.04806148035441163 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.08375177385195182 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.08887966830195516 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2425538811931786 train acc 15/16 valid acc 15/16
Epoch 27 loss 1.069487156930601 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.035467112776697216 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.6257723373327811 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.3095256131389662 train acc 14/16 valid acc 15/16
Epoch 27 loss 0.6288006247371883 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.28257417518836514 train acc 14/16 valid acc 15/16
Epoch 27 loss 0.1371272586742291 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.04224333320359423 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.5116069103163754 train acc 14/16 valid acc 15/16
Epoch 27 loss 0.14508455039448828 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.08056051247041453 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.37971002184251446 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.04136015526873996 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.1746828258860308 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2375516879822821 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2375363637981891 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.417990160149028 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.13447652031933224 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.02121773392722917 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.6250818358771464 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.03901838123111978 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.0580117454037259 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.3331445872435149 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.13997324932089472 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.08175833495249274 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2526858239321104 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.020180074978496842 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2699675591550894 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.3889982896517828 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.15399492457204608 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.08996791245940214 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.547108410386666 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.273454897402476 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.024103133709625666 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.0880112393198008 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.2602618234988395 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.017029387533552365 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.38080997457929266 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.18284254032403732 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.24758474556860793 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.09488344395458298 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.013940500915953963 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.051397724200156966 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.34039734896674995 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.14354675979743645 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.07303521054547223 train acc 16/16 valid acc 15/16
Epoch 27 loss 1.0335692771916132 train acc 15/16 valid acc 15/16
Epoch 27 loss 0.10314390064204637 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.00854064052161934 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.09550994335945504 train acc 16/16 valid acc 15/16
Epoch 27 loss 0.04744383242242019 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.003925029201180453 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.25080048127530347 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.12031444276343597 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10455338921691061 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.023062279128280985 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.06178179241252947 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10004084545475386 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.11989910287336589 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.20159447935241087 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.052510757509171405 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.18065948398062978 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.043037837323884845 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.01752596242097423 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.021850683562855664 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.14787604583097355 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.5287857070742504 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.03025252947951627 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.0936800372491188 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.034047037589102956 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.02050514835290182 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.30080588286899446 train acc 15/16 valid acc 14/16
Epoch 28 loss 0.581238165386114 train acc 15/16 valid acc 16/16
Epoch 28 loss 0.4314310339429418 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.07422786514522411 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.07592748497877258 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10591542552353865 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.16079575247894295 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.07358522739730558 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.2175303417715722 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.06044640479857563 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10257488854784406 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.32964265761746947 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.04606715898661824 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.5606494750724903 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.4387058481063104 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.17792152725855598 train acc 16/16 valid acc 16/16
Epoch 28 loss 0.47750195342093577 train acc 16/16 valid acc 16/16
Epoch 28 loss 0.5148739342426337 train acc 16/16 valid acc 14/16
Epoch 28 loss 0.6868383908235617 train acc 15/16 valid acc 16/16
Epoch 28 loss 0.16366609891486628 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.018931045962618055 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.14494784927164744 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.1799752432343298 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.13961314435814406 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.13594841080095932 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.11467593183552838 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.12787861701476827 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.19998068335582633 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.008408630657721829 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.13135717711713957 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10568688612352706 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.2933645560998773 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.12158356807325192 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.14502591435032874 train acc 16/16 valid acc 16/16
Epoch 28 loss 0.4335682367432421 train acc 14/16 valid acc 15/16
Epoch 28 loss 0.22662353164594604 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.4045592941238131 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.10371370119797418 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.3380070450650505 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.17258640310613169 train acc 16/16 valid acc 15/16
Epoch 28 loss 0.3081811858477564 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.7983878703019505 train acc 15/16 valid acc 15/16
Epoch 28 loss 0.3936668778976571 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.013082733686771238 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.7030166867762924 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3224956188169126 train acc 16/16 valid acc 14/16
Epoch 29 loss 0.3967239996148479 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.03377696400805829 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.04901065004681919 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.27464580515076314 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.04298043083295569 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.029018824672511805 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.32998458842262834 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.1777106784452737 train acc 16/16 valid acc 16/16
Epoch 29 loss 0.43627980726717813 train acc 16/16 valid acc 16/16
Epoch 29 loss 0.08337885311725013 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.5838101430292706 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.13522881179677007 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.24612747983856334 train acc 15/16 valid acc 14/16
Epoch 29 loss 0.16274364629166438 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.12098010739428866 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3254137077298802 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.1606954456401841 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.6899900205421785 train acc 15/16 valid acc 15/16
Epoch 29 loss 0.46153088223309224 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.20383116130252046 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.05241740169136121 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.48759567342732557 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3027756325916263 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.11537964281853978 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.06322336583327576 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.0834187490828348 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.19162518198330122 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.082226712226586 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.11783677244246275 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.23779165995733412 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.29308331571660234 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.26077794072272875 train acc 15/16 valid acc 15/16
Epoch 29 loss 0.07103436295662006 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.16111587070123484 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.13205598753705503 train acc 15/16 valid acc 15/16
Epoch 29 loss 0.013642214423610804 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.008217373279469267 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.025505412538074645 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.0904900125535733 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.026051791423952864 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.008740001036811652 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.25753847784577066 train acc 14/16 valid acc 14/16
Epoch 29 loss 0.019978102919868096 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3201649887166791 train acc 15/16 valid acc 15/16
Epoch 29 loss 0.19383548858062202 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.48736349054990924 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.03544681903689883 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3830871321065531 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.5143444368784822 train acc 15/16 valid acc 15/16
Epoch 29 loss 0.3303171135405852 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.22996693187011638 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.45037648011805687 train acc 14/16 valid acc 15/16
Epoch 29 loss 0.12285586841093947 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.3544089897009544 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.1635826239017705 train acc 16/16 valid acc 16/16
Epoch 29 loss 0.06276138212385655 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.18949586312191702 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.2730293014198613 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.13634714540416917 train acc 16/16 valid acc 15/16
Epoch 29 loss 0.2558377477788565 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.0003287014489261156 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.25428120223535117 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.46463183110892375 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.06232622353942313 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.04397257424156791 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.007654334362970361 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.0840540933148528 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.1350557975469513 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.10466983921573064 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.006832786993475674 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.4341135609645956 train acc 16/16 valid acc 15/16
Epoch 30 loss 1.0323467113753872 train acc 16/16 valid acc 16/16
Epoch 30 loss 0.19252426467583494 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.07659079324141295 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.17574134482154913 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.17934980692831726 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.47807598695349623 train acc 14/16 valid acc 15/16
Epoch 30 loss 0.4248674961057456 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.7060992146014806 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.10330861436353198 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.18465462290465015 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.3503616226508681 train acc 15/16 valid acc 14/16
Epoch 30 loss 0.19925288761079618 train acc 16/16 valid acc 14/16
Epoch 30 loss 0.11748368831638421 train acc 16/16 valid acc 14/16
Epoch 30 loss 0.3479734688438235 train acc 14/16 valid acc 15/16
Epoch 30 loss 0.15497122485792647 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.17980534669493017 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.13632356087357098 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.1068155054056445 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.3666573129136378 train acc 16/16 valid acc 16/16
Epoch 30 loss 0.19652824289793977 train acc 15/16 valid acc 16/16
Epoch 30 loss 0.18133817980225292 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.0633211227251291 train acc 16/16 valid acc 15/16
Epoch 30 loss 1.0884546574365088 train acc 15/16 valid acc 14/16
Epoch 30 loss 0.08524598785359622 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.48867748782230347 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.2350043267423293 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.5514488405666584 train acc 16/16 valid acc 16/16
Epoch 30 loss 0.13504960517054698 train acc 16/16 valid acc 16/16
Epoch 30 loss 0.2035446762754023 train acc 16/16 valid acc 16/16
Epoch 30 loss 0.056904196107022666 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.2788566787497661 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.029146188259501937 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.10543313733898994 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.28275677675569943 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.01398812192660796 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.23038153372237047 train acc 15/16 valid acc 15/16
Epoch 30 loss 0.15886969161928644 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.2568474471234502 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.011870645706970545 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.18363455531331666 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.03489978317869073 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.31849814730362525 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.11493771188039864 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.43816757583401167 train acc 14/16 valid acc 15/16
Epoch 30 loss 0.3441785380644074 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.43839352086418043 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.39690038971960395 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.05469974885786564 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.31986049152920265 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.4228796278944027 train acc 14/16 valid acc 15/16
Epoch 30 loss 0.17941300602395857 train acc 16/16 valid acc 15/16
Epoch 30 loss 0.16495722834596943 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.0008289658280649759 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.11026291356407168 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.12120518424734658 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.05376781109831397 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.030975473632945113 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.2409146516083224 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.13986283377004083 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.5693366178928985 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.054219309894400644 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.025323457970073664 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.09073681251446603 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.11360707841263024 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.3029866762668134 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.21929411840513385 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.1921018449174593 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.4019606831093283 train acc 14/16 valid acc 15/16
Epoch 31 loss 0.16076449026951442 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.790922221741931 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.3150453896041602 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.20759866455609732 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.02443899977335142 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.006942359170974624 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.2087778516933332 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.019231672694166944 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.1074769664165506 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.08265990879330264 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.18450608305903146 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.22046552864256316 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.0827456875386768 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.038419402629954816 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.03453713695121062 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.05354071830630392 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.09691890263123373 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.3490532927904191 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.032521436389049675 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.07225413700171074 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.42168353767552985 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.26431076059188086 train acc 15/16 valid acc 16/16
Epoch 31 loss 0.30568180918164345 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.28201133984595894 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.01263098834263191 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.12150955038269791 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.020053529435006805 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.06586609099988927 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.04840698473733533 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.025284873121670264 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.2570891773658406 train acc 15/16 valid acc 16/16
Epoch 31 loss 0.5385695119730989 train acc 15/16 valid acc 16/16
Epoch 31 loss 0.9174162650445855 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.37894603060919696 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.35469034461771837 train acc 16/16 valid acc 16/16
Epoch 31 loss 0.41057286024649065 train acc 15/16 valid acc 15/16
Epoch 31 loss 0.10641042356895564 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.5323279065401112 train acc 16/16 valid acc 13/16
Epoch 31 loss 0.21680813756970901 train acc 16/16 valid acc 14/16
Epoch 31 loss 0.07169055152023877 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.10708001124613073 train acc 16/16 valid acc 14/16
Epoch 31 loss 0.19780179374668577 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.1789393238493924 train acc 16/16 valid acc 13/16
Epoch 31 loss 0.5052188861936289 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.20558242278319128 train acc 15/16 valid acc 16/16
Epoch 31 loss 0.21463381343816262 train acc 16/16 valid acc 15/16
Epoch 31 loss 0.002668298158023953 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.0006040122444102716 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.28593206035018937 train acc 15/16 valid acc 15/16
Epoch 32 loss 0.16498778309271234 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.12425399170580095 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.0008256823194081191 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.4450075764073447 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.14444989428748498 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.67665365356304 train acc 15/16 valid acc 16/16
Epoch 32 loss 1.5473678444433958 train acc 15/16 valid acc 14/16
Epoch 32 loss 0.300019627091555 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.21043582110465753 train acc 15/16 valid acc 15/16
Epoch 32 loss 0.10946320995429085 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.11109444687228129 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.00301860045058073 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.1408490004593181 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.44792884031030833 train acc 15/16 valid acc 15/16
Epoch 32 loss 0.35976512529009597 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.040134588073565805 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.26459363170599876 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.17797877778203644 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.023977972297750463 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.01106669426470095 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.031940593742605904 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.06611598405671984 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.28713696200618233 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.04695494292979953 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.07551121755953902 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.08791686529369364 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.1877724913240729 train acc 16/16 valid acc 15/16
Epoch 32 loss 1.0765589510402416 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.017346896656363128 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.03374025425677895 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.027324416907675064 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.0811672955372462 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.2278913633279526 train acc 15/16 valid acc 15/16
Epoch 32 loss 0.05647946336007259 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.13471517362487564 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.21284730923760883 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.014016044226857772 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.052668129557564954 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.19405440268975027 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.058517942593229684 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.0044246555691765 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.6066044760440361 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.3916718295440087 train acc 16/16 valid acc 14/16
Epoch 32 loss 0.2112500415492342 train acc 16/16 valid acc 14/16
Epoch 32 loss 0.05022903250076895 train acc 16/16 valid acc 14/16
Epoch 32 loss 0.2900849043123201 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.2554025933387418 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.16511051533552065 train acc 15/16 valid acc 16/16
Epoch 32 loss 0.0864225284941502 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.09013358479974103 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.018817134791553808 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.1799673499092158 train acc 15/16 valid acc 16/16
Epoch 32 loss 0.048647875005175714 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.015663992157714832 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.6541436833614606 train acc 16/16 valid acc 15/16
Epoch 32 loss 0.3999906027215662 train acc 15/16 valid acc 13/16
Epoch 32 loss 0.10484085971816379 train acc 16/16 valid acc 13/16
Epoch 32 loss 0.24866325294974695 train acc 16/16 valid acc 14/16
Epoch 32 loss 0.029294064112353125 train acc 16/16 valid acc 14/16
Epoch 32 loss 0.09191003307169537 train acc 16/16 valid acc 16/16
Epoch 32 loss 0.40092158908879016 train acc 15/16 valid acc 16/16
Epoch 33 loss 0.009130331810980358 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.19791453844114926 train acc 15/16 valid acc 14/16
Epoch 33 loss 0.4072272591455073 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.29321890222898955 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.2082352407838303 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.14085807130144595 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.043296163262376786 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.08918415797659197 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.015873080864012117 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.02558914783133862 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.1137232380755025 train acc 15/16 valid acc 16/16
Epoch 33 loss 0.32977698663825367 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.649051448741202 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.04301047224240097 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.2864693889022684 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.23757824287460677 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.07728348743333452 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.40926816947972045 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.46213970669447746 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.4919215927811581 train acc 14/16 valid acc 15/16
Epoch 33 loss 0.08495681671410803 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.3664824810218528 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.00430249002722824 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.01766181767248868 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.42910166173211406 train acc 14/16 valid acc 15/16
Epoch 33 loss 0.38681002254136587 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.08602278366756706 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.3358454177780636 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.20750140044608423 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.02306991066926488 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.02708300002078234 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.20196674780468063 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.018209114315929774 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.17718666382704093 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.15698233112346327 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.13613236375632962 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.011780820504339413 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.09238785838383194 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.007150155360291921 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.009990439337982234 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.056822658938309144 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.01748897328744428 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.027194527879593342 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.006211874551517793 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.030004952911247468 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.007140472962791099 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.17970795778001586 train acc 15/16 valid acc 15/16
Epoch 33 loss 0.12504736772167557 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.012106575700895109 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.008949198667244744 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.09037595905955387 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.17617254786033948 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.0494520286088921 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.05416114599971372 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.15652940287791203 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.012390880454126231 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.15259001315623297 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.019396486936746962 train acc 16/16 valid acc 15/16
Epoch 33 loss 0.4824572842405599 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.05968465472836208 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.009387288473506917 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.10945750102181906 train acc 16/16 valid acc 16/16
Epoch 33 loss 0.18555315683150475 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.0052614151008646606 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.6610175597774696 train acc 15/16 valid acc 13/16
Epoch 34 loss 0.6958321956119526 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.07077156069024895 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.6092604301829843 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.1832108763187032 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.0329663343510049 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.4686061369391028 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.04457962268794724 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.021115598369387877 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.1933527847984692 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.0966770479008417 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.053125846570808016 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.03901805491385011 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.4873997721384683 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.2378922016124184 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.30238759324638376 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.06455719951007415 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.5143227815404582 train acc 15/16 valid acc 14/16
Epoch 34 loss 0.3746275067595931 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.0804395510304589 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.10234721644317878 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.25935647262686906 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.07421528216880252 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.012098598561311614 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.24580733297929752 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.009860124094573638 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.1753454431729942 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.07221883298026484 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.004421194355391812 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.019220946527419246 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.0028715039407954095 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.16510783604986187 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.048933108983754484 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.06500523492873608 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.2332405752178106 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.08498225958255139 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.41178817011645896 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.12546139137353046 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.24869071565467155 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.0736947200565915 train acc 15/16 valid acc 14/16
Epoch 34 loss 0.2505693646924707 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.06283162665452308 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.25363356126328773 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.5737575886987947 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.029627688438870228 train acc 16/16 valid acc 16/16
Epoch 34 loss 0.2058709122482545 train acc 14/16 valid acc 15/16
Epoch 34 loss 0.5142752856483165 train acc 14/16 valid acc 16/16
Epoch 34 loss 0.7670685826657347 train acc 15/16 valid acc 15/16
Epoch 34 loss 0.3543556617407731 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.46451901551749186 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.13726800407886333 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.016154557698806634 train acc 16/16 valid acc 15/16
Epoch 34 loss 0.07642145185388984 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.14092667017170418 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.03714748425442212 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.4464501893315663 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.388601942410901 train acc 15/16 valid acc 14/16
Epoch 34 loss 0.02357659110580803 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.05153910995323892 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.028365546738961224 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.3652259912437611 train acc 16/16 valid acc 14/16
Epoch 34 loss 0.09677489480148704 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.41393270830407203 train acc 15/16 valid acc 14/16
Epoch 35 loss 0.21801146169345964 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.10198344664237433 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.03752136039238506 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.14366322960019806 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.1240360216371679 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.25894194725152064 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.16271460485184672 train acc 15/16 valid acc 14/16
Epoch 35 loss 0.26640510085497077 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.03766462313781303 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.2646591955362029 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.5966513939388968 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.017190302728478782 train acc 16/16 valid acc 15/16
Epoch 35 loss 1.3571617870876673 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.3645476637228278 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.2471740704593191 train acc 15/16 valid acc 14/16
Epoch 35 loss 0.3611222759654238 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.25355730935868087 train acc 15/16 valid acc 14/16
Epoch 35 loss 0.10484835776974298 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.038275549024258254 train acc 16/16 valid acc 15/16
Epoch 35 loss 2.19687017619622 train acc 14/16 valid acc 13/16
Epoch 35 loss 0.4191676547628931 train acc 15/16 valid acc 16/16
Epoch 35 loss 0.14791061675798636 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.10031991646696885 train acc 16/16 valid acc 14/16
Epoch 35 loss 0.11726197855595125 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.18892876871860667 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.28125328129523136 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.14321470542470718 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.18778398672027724 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.02580499972263804 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.007423378018005221 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.33604761401869887 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.005478230285837054 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.15847959198754524 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.21248082099057558 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.1037016157809567 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.1635460946648533 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.03236083645085044 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.17007638029953748 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.2562372804793538 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.0319791655046768 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.1685267814295825 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.040202241317325305 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.047842370663878364 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.13695996522053666 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.0842387903680738 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.27691815500304356 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.24379524444679168 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.19496249565190635 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.5360173665121057 train acc 16/16 valid acc 16/16
Epoch 35 loss 0.705775648084725 train acc 15/16 valid acc 16/16
Epoch 35 loss 0.2690609376734931 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.13713326152368785 train acc 16/16 valid acc 16/16
Epoch 35 loss 0.18568949420220182 train acc 15/16 valid acc 15/16
Epoch 35 loss 0.5214748331908978 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.32800721328732607 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.0760593865999883 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.10137910857209417 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.366791729052489 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.06321404682035302 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.13985519862211657 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.257004478151722 train acc 16/16 valid acc 15/16
Epoch 35 loss 0.13399036642779139 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.011750247323419435 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.3958967693181746 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.5338628057450484 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.4332853975097549 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.45746939149895377 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.07666734166698425 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.22087126014234432 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.33042719950603433 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.2753731630659867 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.013234247196079288 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.08032664494285305 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.15279474607679291 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.37193571496225875 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.6274825658458898 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.09534807202256616 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.16764933853815062 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.6585732875669067 train acc 13/16 valid acc 14/16
Epoch 36 loss 0.43739848455666486 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.4148995040100067 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.34744999895272044 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.3871786504888851 train acc 15/16 valid acc 14/16
Epoch 36 loss 0.4740031629812776 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.028544513898793953 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.20305580944323176 train acc 15/16 valid acc 14/16
Epoch 36 loss 0.2470691511046599 train acc 15/16 valid acc 14/16
Epoch 36 loss 0.07455689696762245 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.08861707735114278 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.0583177905345969 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.39730823065105203 train acc 14/16 valid acc 14/16
Epoch 36 loss 0.17467774535648442 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.24747201678637004 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.4528924994507366 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.16549233220250548 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.08324893377166245 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.08547663254023363 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.2123992481217338 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.11456910379065258 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.08111308913437702 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.03580577853880049 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.20947204616916165 train acc 15/16 valid acc 16/16
Epoch 36 loss 0.0036633326240573614 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.01243449702059629 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.07772776115623974 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.1139146802324196 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.03346714452692007 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.014608612595370912 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.30161453978527675 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.09872398698020435 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.22211424545209674 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.28291694550107277 train acc 15/16 valid acc 15/16
Epoch 36 loss 0.09144561990886071 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.25212577560168814 train acc 15/16 valid acc 14/16
Epoch 36 loss 0.03680431654078231 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.005855638456524441 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.05442180495556287 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.016880041567622794 train acc 16/16 valid acc 14/16
Epoch 36 loss 0.6699641650269668 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.18175509043353552 train acc 16/16 valid acc 16/16
Epoch 36 loss 0.03352735223495714 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.10381758172642955 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.013068109307497142 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.05395405730312092 train acc 16/16 valid acc 15/16
Epoch 36 loss 0.8649594585382587 train acc 14/16 valid acc 15/16
Epoch 37 loss 0.0031640463126627904 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.27456480909898534 train acc 15/16 valid acc 15/16
Epoch 37 loss 0.04266262510219917 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.15446572536655703 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.0036238477880301285 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.007062286549446362 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.032915347227130444 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.006055537844860907 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.06753903034064265 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.06010488877325276 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.20794771386271504 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.32456367106429246 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.017614501326418015 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.16782458841010414 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.0554126869495954 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.1941923039178065 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.2104483914215849 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.035530390121929746 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.008548955668831106 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.05530089614563907 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.265223475437953 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.011184297963272052 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.1399790221823901 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.25799206050181833 train acc 15/16 valid acc 15/16
Epoch 37 loss 0.015213006308777977 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.18998364111220567 train acc 15/16 valid acc 15/16
Epoch 37 loss 0.2168554617666527 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.5824866234397427 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.06405310125201495 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.09740306276310924 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.17428748293257046 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.08352258343993867 train acc 15/16 valid acc 15/16
Epoch 37 loss 0.07029186453934921 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.1080757603722592 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.36699608530192507 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.008225322291281057 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.07705495715817068 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.5177775184346869 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.06381453562464692 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.09055654077480356 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.038687423364519746 train acc 16/16 valid acc 16/16
Epoch 37 loss 0.22955502411299894 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.03473300969146387 train acc 16/16 valid acc 15/16
Epoch 37 loss 0.2512545967766615 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.19595974226314417 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.006232069121612787 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.12655242200010555 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.5432039065482797 train acc 14/16 valid acc 15/16
Epoch 37 loss 0.18187247325671013 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.0073571405547095845 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.14670099991725938 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.08184719352016104 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.01027672325134684 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.03608329353590349 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.05115468950466999 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.10223780164914635 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.20909565911100897 train acc 15/16 valid acc 14/16
Epoch 37 loss 0.4172179361635939 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.03044840963217236 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.18350326156781036 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.055928174517097744 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.03003949995833901 train acc 16/16 valid acc 14/16
Epoch 37 loss 0.03469029522105332 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.07531581601376781 train acc 15/16 valid acc 14/16
Epoch 38 loss 0.09700168618961544 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.021433114028565438 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.060296654199113016 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.00014241303831153147 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.8295843390955525 train acc 15/16 valid acc 14/16
Epoch 38 loss 0.3900252070835398 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.3652893357382061 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.796696329627193 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.020578187168308757 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.034910411336544316 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.5944373459020246 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.11812626197301572 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.11005284072485211 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.010690487953879607 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.333439187085976 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.0395770604910299 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.2941541426392552 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.09386641197316849 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.004440751773261682 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.15885268566449728 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.07784551978215841 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.13199261823775377 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.12153657470937251 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.6748113868476411 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.03987276813418442 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.14678254652808026 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.04240808276315147 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.27667324111670055 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.004553578288259007 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.0265915033490599 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.5254855919945476 train acc 15/16 valid acc 14/16
Epoch 38 loss 0.02198078082190662 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.014631203913688082 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.2475504552400376 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.34145659882829343 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.07517501490096337 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.09321002214803507 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.002296223827903844 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.035423232398369746 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.012070863712275214 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.2117670463143362 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.1420279253134112 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.007561865491653846 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.20750683125781194 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.005862213862419402 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.10219644720878499 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.11753429126970355 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.0020842785346546116 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.08435280803226969 train acc 16/16 valid acc 14/16
Epoch 38 loss 0.23497232912697055 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.27772665073833075 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.3372237458254443 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.24093079932511083 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.19026672553334745 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.09487096640556142 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.04804204074378848 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.26812466429994125 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.011598550515348437 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.2213439161651718 train acc 15/16 valid acc 15/16
Epoch 38 loss 0.03943517118851659 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.22630550267125357 train acc 16/16 valid acc 15/16
Epoch 38 loss 0.15069691062830923 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.00038257906941026445 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.14247849366747187 train acc 15/16 valid acc 15/16
Epoch 39 loss 0.055150827784720934 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.07974706103675894 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.21258397795142023 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.2160466628168476 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.1667793128511619 train acc 15/16 valid acc 16/16
Epoch 39 loss 0.10312337228094992 train acc 16/16 valid acc 16/16
Epoch 39 loss 0.08926688525646691 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.06032397314352383 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.0023641035646867602 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.04588532327827054 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.0007103604845694975 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.06948716115193874 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.2612191154460067 train acc 16/16 valid acc 16/16
Epoch 39 loss 0.47585471670739055 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.6815950262989794 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.21853636478757743 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.0941833891544725 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.1614772740182971 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.6531941648071049 train acc 15/16 valid acc 15/16
Epoch 39 loss 0.1067926105907822 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.011050377867429004 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.018248288616773942 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.04310737254023779 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.07485086999526863 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.09502889525093061 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.051782783285814316 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.04400504869244243 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.005569713717640763 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.6014279941793206 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.16892946031315892 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.143086177796739 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.23572257769034527 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.15771118779887658 train acc 15/16 valid acc 15/16
Epoch 39 loss 0.06784622574708872 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.02751269727642287 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.005375934184493814 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.21341033356435207 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.014769187239308462 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.09108630534114373 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.07008626749857286 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.1967740666099716 train acc 14/16 valid acc 14/16
Epoch 39 loss 0.5228400110660486 train acc 15/16 valid acc 15/16
Epoch 39 loss 0.014030349215155006 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.11841983583857099 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.06531379979788615 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.17517148584140407 train acc 15/16 valid acc 15/16
Epoch 39 loss 0.07017058496347409 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.07537680378577151 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.20542809326318487 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.14000305018920764 train acc 16/16 valid acc 16/16
Epoch 39 loss 0.05479831532383857 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.02588618084483067 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.07515176341841465 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.5696876412845269 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.03844550738943269 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.023452347768921383 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.8321362911167832 train acc 15/16 valid acc 14/16
Epoch 39 loss 0.4431669879398115 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.053517560688612895 train acc 16/16 valid acc 14/16
Epoch 39 loss 0.044071133098245965 train acc 16/16 valid acc 15/16
Epoch 39 loss 0.17373547927613558 train acc 15/16 valid acc 15/16
Epoch 40 loss 3.588704757759683e-05 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.21941266449537192 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.5944988335194195 train acc 16/16 valid acc 13/16
Epoch 40 loss 0.014218819179529095 train acc 16/16 valid acc 14/16
Epoch 40 loss 0.3077486263855091 train acc 16/16 valid acc 14/16
Epoch 40 loss 0.049091337138352134 train acc 16/16 valid acc 14/16
Epoch 40 loss 0.1230636166254639 train acc 15/16 valid acc 14/16
Epoch 40 loss 0.30717294741460793 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.48800159217813915 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.5633031902754347 train acc 16/16 valid acc 14/16
Epoch 40 loss 0.38608272004186994 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.6007232617077402 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.021285787153629426 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.1158762021304381 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.09047947570546891 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.204366205916871 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.18056146397737352 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.15251539516308943 train acc 15/16 valid acc 14/16
Epoch 40 loss 0.9191467598750952 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.21553946478429206 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.11505930818471301 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.11536413496206 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.6080059985018461 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.1531417322413514 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.4081240935137471 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.7597129117980909 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.12960838091460558 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.035258673393039196 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.09086695414575131 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.02866347309975935 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.11447243803925664 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.09183667977626826 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.06556942365625289 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.240672154186564 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.21258555317473904 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.043346939056911823 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.05024406803155641 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.2743663160172788 train acc 15/16 valid acc 15/16
Epoch 40 loss 0.011758414115790435 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.01998359960801183 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.07004234097506012 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.0316152609696021 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.06914916236950662 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.01366464602222585 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.13354330431700576 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.001661274399068163 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.0804899677198962 train acc 15/16 valid acc 16/16
Epoch 40 loss 0.038510548380058594 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.13172178626928763 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.01247105647898486 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.017678321341662875 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.021280217858759835 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.39505837850626474 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.20295643569566296 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.04750819465143616 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.06345542021056873 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.5194838446007963 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.12644838949977197 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.45314973062531205 train acc 14/16 valid acc 14/16
Epoch 40 loss 0.2549476902233869 train acc 16/16 valid acc 15/16
Epoch 40 loss 0.18404508390686788 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.0640698610650543 train acc 16/16 valid acc 16/16
Epoch 40 loss 0.22839770999983827 train acc 15/16 valid acc 15/16
Epoch 41 loss 1.9388557050988305e-06 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.32080506040075174 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.1348676044012168 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.008471405605250863 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.39753988358353454 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.18122497283157468 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.06698470263137105 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.4070858647279151 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.021614900218642357 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.011207264504378074 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.06607068376788625 train acc 15/16 valid acc 16/16
Epoch 41 loss 0.014862383469923504 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.32654164472552305 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.19427028046410297 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.0738094035327303 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.15394410747925583 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.5442097151329421 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.07945939616034252 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.1667455938864955 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.1300121293828412 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.03585695349999976 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.008786603297970309 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.006449858559364086 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.27064679394662283 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.11409656983134382 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.13643962268586646 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.02576264349325428 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.1412325432424868 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.5855882071509189 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.2721365080516381 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.3199553012874374 train acc 15/16 valid acc 16/16
Epoch 41 loss 0.028818928318531434 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.5012755687194761 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.12418494625284564 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.35573069795476026 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.12022649982120599 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.04711419655413127 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.4867209099931375 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.014793836215906374 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.15180883481644164 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.026890649909855114 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.5006351740668403 train acc 15/16 valid acc 16/16
Epoch 41 loss 0.33299514805980757 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.011706232084392944 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.16021431157931773 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.015523174091251039 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.255706588626778 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.09744312555786674 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.2817539223659979 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.5726323144881635 train acc 14/16 valid acc 15/16
Epoch 41 loss 0.2867012995166998 train acc 16/16 valid acc 16/16
Epoch 41 loss 1.0122551478662432 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.6410525359568298 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.27646779194882254 train acc 15/16 valid acc 15/16
Epoch 41 loss 0.5708693020186757 train acc 14/16 valid acc 15/16
Epoch 41 loss 0.40299398250567103 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.2169017029746355 train acc 15/16 valid acc 16/16
Epoch 41 loss 0.03290008189467259 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.015959256565887005 train acc 16/16 valid acc 16/16
Epoch 41 loss 0.47767236635551674 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.20790622042496998 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.40705627552924617 train acc 16/16 valid acc 15/16
Epoch 41 loss 0.08205863243964417 train acc 16/16 valid acc 16/16
Epoch 42 loss 0.12394788163625764 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.19155472137891622 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.4807903763234341 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.11069959020740042 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.2041670632529973 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.01837194025981037 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.3355840600607527 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.32454983081654165 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.5015515725595944 train acc 14/16 valid acc 16/16
Epoch 42 loss 0.19190567155220278 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.6546415042690722 train acc 15/16 valid acc 16/16
Epoch 42 loss 0.5642295673548361 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.06964938788655865 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.1078218619013416 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.5576501295077888 train acc 14/16 valid acc 15/16
Epoch 42 loss 0.20955655025711287 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.24563373656798765 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.29071906437264255 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.008084713372302555 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.16974586398657276 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.0534910967084977 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.21036041399050617 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.024669589590819593 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.2026075410390122 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.001682934900097216 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.0838646858469119 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.013720044425143994 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.002255951357130433 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.021051980183644042 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.10047901035672084 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.014303500114932357 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.07792223409962866 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.010961253210464894 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.05526836263800615 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.8343318289892943 train acc 14/16 valid acc 15/16
Epoch 42 loss 0.07158204815800469 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.19673206982167557 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.12020229237172615 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.007963584446360333 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.07387535759077178 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.008050453266944913 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.16972349223564015 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.03672923804851916 train acc 16/16 valid acc 15/16
Epoch 42 loss 1.1045108519218567 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.1030211303960776 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.018472991812548214 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.1305734225702625 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.04986504571135146 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.43725315383463137 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.015072792917387308 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.09161618587865156 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.020269610279910668 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.24186014692169056 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.007431158827626205 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.3209769814051065 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.05689291654409781 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.3056961382363217 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.3235403150917658 train acc 15/16 valid acc 15/16
Epoch 42 loss 0.0849866447884531 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.19291778096340773 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.3453758699185013 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.034681211316468744 train acc 16/16 valid acc 15/16
Epoch 42 loss 0.28655773597217066 train acc 15/16 valid acc 16/16
Epoch 43 loss 5.622861618297111e-06 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.10254648728409055 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.10861134089223176 train acc 15/16 valid acc 15/16
Epoch 43 loss 0.3592529960588076 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.024002477457599772 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.013797788428619721 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.05938514902175501 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.24856389867009387 train acc 15/16 valid acc 15/16
Epoch 43 loss 0.0899470591945477 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.0023532996022998867 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.07366932171079796 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.2262652509025284 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.01588699415869554 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.12783619927790546 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.2366427058205317 train acc 15/16 valid acc 15/16
Epoch 43 loss 0.16745958928579877 train acc 15/16 valid acc 15/16
Epoch 43 loss 0.10671547486866516 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.08728558179102516 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.07806509609060217 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.03207929985251785 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.1545271553660542 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.16756798925662555 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.18572516125317298 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0808857008456112 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.005515247422445564 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.004829537163973577 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.13421007468467583 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.004971730707736646 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0661443547405875 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.03833748695804357 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.6552987891669954 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.22720723129526407 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.05315850676494541 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.21098528078348627 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.025643999986291875 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.20665543344218806 train acc 15/16 valid acc 16/16
Epoch 43 loss 0.03320922221865032 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.09714055746212075 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.20573859776005304 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0033161459448485253 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0284536470119765 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.02397817136002181 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.2924180465973177 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.005879849665657131 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.517132086809344 train acc 15/16 valid acc 16/16
Epoch 43 loss 0.019854280724865385 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.40990077972155387 train acc 16/16 valid acc 16/16
Epoch 43 loss 0.18775239364934987 train acc 15/16 valid acc 15/16
Epoch 43 loss 0.025744654720469613 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.05119099260839692 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.3460071284078534 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0634986540653793 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.04461842392110511 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.010192237124423567 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.04029678609855026 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.07832633852400717 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.0960148587038401 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.11920951787699409 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.38439727774112775 train acc 15/16 valid acc 14/16
Epoch 43 loss 0.5181150046632383 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.32116902052018825 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.15110749878099333 train acc 16/16 valid acc 15/16
Epoch 43 loss 0.8016737259262271 train acc 14/16 valid acc 15/16
Epoch 44 loss 6.256977355049147e-05 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.15592050613530278 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.03200109479227017 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.1599352241229742 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.5887777949729445 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.05813956038528681 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.1841767907781444 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.5325109277241401 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.13980167910228886 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.11650241415219756 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.09018549503875896 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.08061680138318283 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.09207096346978405 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.3012903637247597 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.011960529637438948 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.24992418335461947 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.047325752904601295 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.0933120902256036 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.9603957858013167 train acc 16/16 valid acc 16/16
Epoch 44 loss 0.08220327085441191 train acc 16/16 valid acc 16/16
Epoch 44 loss 0.6518794418662247 train acc 15/16 valid acc 14/16
Epoch 44 loss 0.14551969371907184 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.017980205908631565 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.4445169800694287 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.01348185652103045 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.1954225235889583 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.27305602288584296 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.004822233774137465 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.015403694090475123 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.5123595103745704 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.03968434882354136 train acc 16/16 valid acc 14/16
Epoch 44 loss 0.0999100356796631 train acc 15/16 valid acc 16/16
Epoch 44 loss 0.1493778141377192 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.2531232064196862 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.6543471475841234 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.14103528148267816 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.18426221080461325 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.023455362336195953 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.019826583590981803 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.10914503950441515 train acc 16/16 valid acc 16/16
Epoch 44 loss 0.0025258035747862946 train acc 16/16 valid acc 16/16
Epoch 44 loss 0.010465325818935495 train acc 16/16 valid acc 16/16
Epoch 44 loss 0.08289239090496493 train acc 15/16 valid acc 16/16
Epoch 44 loss 0.3552160468094252 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.008997742321640148 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.00811252900393118 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.36872652870513173 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.2904064127211572 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.2061136409794862 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.06673036834330148 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.1699259544325729 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.11621849372631667 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.03965504633059694 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.017405140741055935 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.08590027803323663 train acc 15/16 valid acc 15/16
Epoch 44 loss 0.03354951047447592 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.46923769568324025 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.01253775537104277 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.11549669274340303 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.10480355467183519 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.013838845176237846 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.0458449855351655 train acc 16/16 valid acc 15/16
Epoch 44 loss 0.12489115349484589 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.0001324713385107451 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.07409650464954444 train acc 15/16 valid acc 15/16
Epoch 45 loss 1.06787179365108 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.6915848173800117 train acc 14/16 valid acc 16/16
Epoch 45 loss 0.1046668341074862 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.003767835943990984 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.01807185323541409 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.14287543027780544 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.15002811678127279 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.008344131187358803 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.48702078547581273 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.01831914956882756 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.041701620779296344 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.4895356303728313 train acc 15/16 valid acc 16/16
Epoch 45 loss 0.39648014651565866 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.14082942820991898 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.27889137188631374 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.1091256273795017 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.16079927450656237 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.21805372466637454 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.045812099703642604 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.16586588930873 train acc 15/16 valid acc 16/16
Epoch 45 loss 0.010841091806525127 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.3911189568582202 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.01781104080949235 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.1412902946140246 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.5559688332239165 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.014258663616464311 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.2878008472023085 train acc 15/16 valid acc 16/16
Epoch 45 loss 0.11011646553527944 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.09778361342580656 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.008045327351346207 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.04292707101669996 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.020324963047202223 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.00920788484628859 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.2265234416274106 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.21008136221186877 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.08952539030119369 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.005545212548733769 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.00754242759831379 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.0017344822128875742 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.0036992819006612457 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.008012522843780357 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.01211176462066306 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.37030106144617037 train acc 15/16 valid acc 14/16
Epoch 45 loss 0.035318079241106 train acc 16/16 valid acc 14/16
Epoch 45 loss 0.07312166563756296 train acc 15/16 valid acc 14/16
Epoch 45 loss 0.11170218860537547 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.17504406479276938 train acc 16/16 valid acc 14/16
Epoch 45 loss 0.0011708301295115563 train acc 16/16 valid acc 14/16
Epoch 45 loss 0.29168555742177155 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.041106291010950126 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.1674398045427343 train acc 16/16 valid acc 14/16
Epoch 45 loss 0.2935356008256029 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.1729028342448326 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.01315042825870871 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.1035005456726056 train acc 16/16 valid acc 15/16
Epoch 45 loss 0.39451361421768494 train acc 15/16 valid acc 15/16
Epoch 45 loss 0.25459231939206145 train acc 15/16 valid acc 16/16
Epoch 45 loss 0.3870406541369703 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.3771481128707972 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.3041478441893916 train acc 16/16 valid acc 16/16
Epoch 45 loss 0.23664420443182727 train acc 16/16 valid acc 16/16
Epoch 46 loss 8.421903524854699e-06 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.14743618999388855 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.04137440054051113 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.2912720273956057 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.014235826610742323 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.0622850366983802 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.03861210585325586 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.004800425288548521 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.012932462371830149 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.2250960780466047 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.04932695631563327 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.001230501824753747 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.03025438144925369 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.0008717187034432397 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.013115058270314983 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.08087341658200424 train acc 15/16 valid acc 16/16
Epoch 46 loss 0.16176677269561174 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.006362271365655856 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.3206856070416799 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.05736456761235288 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.002019601218176315 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.007297759272237104 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.14126007087586578 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.1046691169145568 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.10617577805102107 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.11729776204110073 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.3515413156247768 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.44388531794838143 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.27098486910277203 train acc 15/16 valid acc 15/16
Epoch 46 loss 0.2676795881831867 train acc 15/16 valid acc 14/16
Epoch 46 loss 0.8854877230458357 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.05902470812288247 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.3586682914415479 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.11559245397323359 train acc 15/16 valid acc 15/16
Epoch 46 loss 0.11571881252135854 train acc 15/16 valid acc 15/16
Epoch 46 loss 0.04814987090130286 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.39618083038363355 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.006768225340024111 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.32784545510317814 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.014489452558324414 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.127387220649586 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.05725676410392456 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.05671682323256481 train acc 16/16 valid acc 15/16
Epoch 46 loss 0.2009367205478083 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.14487769246110388 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.04608859272801176 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.14154006424188845 train acc 15/16 valid acc 15/16
Epoch 46 loss 0.03792134164215447 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.21231659971118738 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.05831945507532921 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.20724606502681234 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.07042232026013939 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.001821965936654236 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.013983034748751453 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.0003641185011142 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.04289789878302502 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.21597835986162794 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.013113869940095857 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.566054336553789 train acc 15/16 valid acc 16/16
Epoch 46 loss 0.05540650134905039 train acc 15/16 valid acc 16/16
Epoch 46 loss 0.006116484610552666 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.002135284840432123 train acc 16/16 valid acc 16/16
Epoch 46 loss 0.07066380216717462 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.00025735255826907097 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.27462569141248094 train acc 15/16 valid acc 16/16
Epoch 47 loss 0.049172455671755225 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.012429592741875274 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.2130782537567051 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.09981821493885742 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.022753129879769594 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.11177110271764935 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.1772257999296991 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.07479206424846543 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.14894103763123726 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.08257525956736009 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.029571340983245785 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.027707138540217613 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.5046091384674986 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.08745636858946154 train acc 15/16 valid acc 15/16
Epoch 47 loss 0.2084148137437487 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.16401517571326873 train acc 14/16 valid acc 15/16
Epoch 47 loss 0.12733050362314072 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.001420939306780215 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.11636722946984596 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.0608756478143028 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.009525726574819461 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.049446296065510854 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.25242129450026224 train acc 15/16 valid acc 16/16
Epoch 47 loss 0.25135057710321035 train acc 15/16 valid acc 15/16
Epoch 47 loss 0.5978808377875592 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.005540682623258646 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.035322109151487195 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.17304776599336072 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.009319933586322982 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.015138963104034477 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.02415035334769124 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.020068802303893726 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.5077113187753362 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.06560719746666886 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.07383624671804324 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.1789135529786895 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.08028665645640211 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.33073670837131935 train acc 15/16 valid acc 15/16
Epoch 47 loss 3.237867730405928e-05 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.015330239358776329 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.00042424502439889655 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.03769752458661353 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.05892361896404975 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.17112836344819785 train acc 16/16 valid acc 16/16
Epoch 47 loss 2.1224702694193387 train acc 16/16 valid acc 14/16
Epoch 47 loss 0.45315398248967564 train acc 15/16 valid acc 14/16
Epoch 47 loss 0.19985275344197723 train acc 15/16 valid acc 15/16
Epoch 47 loss 0.08083340962475553 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.09916509566820823 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.04985332852502296 train acc 16/16 valid acc 14/16
Epoch 47 loss 0.1357960228382732 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.1893243781620994 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.2084625518314634 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.025724689613462207 train acc 16/16 valid acc 16/16
Epoch 47 loss 0.3752793182758043 train acc 16/16 valid acc 14/16
Epoch 47 loss 0.009471754003384682 train acc 16/16 valid acc 14/16
Epoch 47 loss 0.25759390566600693 train acc 15/16 valid acc 15/16
Epoch 47 loss 0.09931861001920649 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.7236332026346062 train acc 15/16 valid acc 15/16
Epoch 47 loss 0.27049279990564956 train acc 16/16 valid acc 15/16
Epoch 47 loss 0.05533446290482577 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.0005066828733587902 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.026134066295961902 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.29187981082852726 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.05116880572819063 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.006808270409069544 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.016232217293937912 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.265195450570925 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.05204828380725765 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.06153574257641043 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.15511504399603057 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.1569607681283662 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.25655647535076664 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.11716024638932541 train acc 15/16 valid acc 16/16
Epoch 48 loss 0.10565228037108425 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.012068996846700377 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.20847343954472855 train acc 15/16 valid acc 15/16
Epoch 48 loss 0.055571208783979226 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.21236023515473038 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.06706594285883516 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.29449622932875263 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.1238877481565926 train acc 15/16 valid acc 16/16
Epoch 48 loss 0.2203364189716886 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.5023912624176196 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.11690817843468251 train acc 15/16 valid acc 15/16
Epoch 48 loss 0.30319841184052104 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.11663379319865272 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.4987784969528198 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.015437533367410348 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.13902426962414166 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.005160222256722677 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.02857788285780248 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.39472393397002437 train acc 15/16 valid acc 15/16
Epoch 48 loss 0.026538297803919357 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.019884038401633525 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.14288523243113793 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.20007730826757406 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.027746580513708256 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.1082564978566902 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.004177125684325245 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.018737377095986957 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.014512664565196757 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.18255357710313272 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.04906466983761469 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.02502705826085705 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.05183754566997726 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.06258717659153783 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.516131361360068 train acc 15/16 valid acc 15/16
Epoch 48 loss 0.22249592914362587 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.08715096730629769 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.001765117421607899 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.24115369220372967 train acc 15/16 valid acc 16/16
Epoch 48 loss 0.07396853817697192 train acc 15/16 valid acc 16/16
Epoch 48 loss 0.18672088936628142 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.039639715595121285 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.011958515067307232 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.0037355612813048614 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.0433722755935742 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.00707211883916359 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.019235115026434782 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.0509795948502242 train acc 16/16 valid acc 15/16
Epoch 48 loss 0.2993902059542307 train acc 16/16 valid acc 16/16
Epoch 48 loss 0.25408959510782253 train acc 15/16 valid acc 15/16
Epoch 48 loss 0.4258199048061496 train acc 15/16 valid acc 16/16
Epoch 49 loss 3.1369682260683446e-05 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.41813486209385164 train acc 15/16 valid acc 15/16
Epoch 49 loss 0.28581203965096524 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.061783593077954425 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.03425858708633324 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.01329465391030189 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.8430484960767721 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.03501292921514265 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.2122682725576973 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.08487390600792168 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.017644884184507954 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.05143618079492839 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.10531725910337307 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.04643903038334284 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.27480292914376264 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.5176109581936668 train acc 15/16 valid acc 16/16
Epoch 49 loss 0.5806694940039752 train acc 14/16 valid acc 15/16
Epoch 49 loss 0.21876940970159706 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.11948431621004525 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.24931825625589887 train acc 15/16 valid acc 15/16
Epoch 49 loss 0.014294903038316981 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.002501595311849844 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.04617567004293877 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.03906281464746492 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.04770358699505072 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.0034477066740274463 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.6900565271803127 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.4734077853034049 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.10553994422720779 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.023309320423130032 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.027451296436618824 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.0744193319542475 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.09556139504178025 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.053073202688022084 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.01713770950872561 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.24687749238065043 train acc 15/16 valid acc 15/16
Epoch 49 loss 0.002710201965546206 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.08589467036475916 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.007894671475994857 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.008991969027430201 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.004658732996928243 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.063622838890645 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.0017396657979443044 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.0011484920917630471 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.009624904127962176 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.061821300623632114 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.4246230337988144 train acc 15/16 valid acc 15/16
Epoch 49 loss 0.2256556476103751 train acc 15/16 valid acc 16/16
Epoch 49 loss 0.3833130948348088 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.04727678328457757 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.009591831576245565 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.08264047610721696 train acc 16/16 valid acc 16/16
Epoch 49 loss 0.10221002174985262 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.24522717532212385 train acc 16/16 valid acc 15/16
Epoch 49 loss 1.1108048016516787 train acc 15/16 valid acc 16/16
Epoch 49 loss 0.11265719374602226 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.07881668852668758 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.05693030500347114 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.30174485495295805 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.0134953973963198 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.012458736124777353 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.009600925710780424 train acc 16/16 valid acc 15/16
Epoch 49 loss 0.025702776600602665 train acc 16/16 valid acc 15/16
Epoch 50 loss 2.9362812444733737e-05 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.14466367163908345 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.19749497365149157 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.13937344711902788 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.11281625518564613 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.3198087166221671 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.0104784372748206 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.12191220958409121 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.27290974858304473 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.207491158430311 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.00304179746828493 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.06847339927429509 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.027527484069817 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.22550887961959243 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.0029649254486774565 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.6353592341890687 train acc 15/16 valid acc 16/16
Epoch 50 loss 0.717680823755512 train acc 16/16 valid acc 14/16
Epoch 50 loss 0.9596108378680757 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.165344610779001 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.171107283176809 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.12990192742603388 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.10027521537704008 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.2704183454643156 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.36820331229674935 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.2946207227110911 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.08941384892327094 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.018387295443664783 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.056327348707410387 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.02083767977469122 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.028316331895874007 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.025987752751620165 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.037683918114915435 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.0886897893171525 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.2671853810853124 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.14234257353009885 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.21072112517273967 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.28380487570027874 train acc 15/16 valid acc 16/16
Epoch 50 loss 0.3692251387047839 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.41488610975719903 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.04016077188537752 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.3504637953035358 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.03482657246083021 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.22538341561730424 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.09455643845987621 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.24275466583517083 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.006682278338956689 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.3563822520282218 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.7920164474852929 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.10593877113633399 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.001263457683533026 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.3764058603854447 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.11373273543752925 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.28194163185318943 train acc 15/16 valid acc 15/16
Epoch 50 loss 0.024114436498321894 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.13650417585367094 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.495307430221149 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.10468613794711035 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.0030587741003841678 train acc 16/16 valid acc 16/16
Epoch 50 loss 0.1591461139998511 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.032820774207529727 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.012761240193451106 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.008414843121682171 train acc 16/16 valid acc 15/16
Epoch 50 loss 0.27916109709855375 train acc 16/16 valid acc 15/16
</pre>
</details>
</pre>
</details>