# V2V: A Deep Learning Approach to Variable-to-Variable Selection and Translation for Multivariate Time-Varying Data
Pytorch implementation for V2V: A Deep Learning Approach to Variable-to-Variable Selection and Translation for Multivariate Time-Varying Data.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset 'Ionization'
```

- inference
```
python3 main.py --mode 'inf'
```

## Citation 
```
@article{Han-VIS20,
	Author = {J. Han and H. Zheng and Y. Xing and D. Z. Chen and C. Wang},
	Journal = {IEEE Transactions on Visualization and Computer Graphics},
	Number = {2},
	Pages = {1290-1300},
	Title = {{V2V}: A Deep Learning Approach to Variable-to-Variable Selection and Translation for Multivariate Time-Varying Data},
	Volume = {27},
	Year = {2021}}

```
