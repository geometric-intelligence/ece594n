# 2D Geometric Shape Classification 

## File & Folder Descriptions
1 - `./dataset` contains the shapes package from [1]. It is modified to generate 49x49 grayscale geometric shapes. 

2 - `./models` includes the trained models. There is a trained VanillaCNN and SO2SteerableCNN from [2].

## Goal

This project compares the rotation invariance of the VanillaCNN and SO2SteerableCNN for 2D geometric shapes. It shows that the steerable CNNs perform well for rotated inputs. 

## Contributor

[Ozgur Guldogan](https://github.com/guldoganozgur)

## References 

[1] El Korchi, Anas, and Youssef Ghanou. "2D geometric shapes dataset–for machine learning and pattern recognition." Data in Brief 32 (2020): 106090.

[2] Cohen, Taco S., and Max Welling. "Steerable cnns." arXiv preprint arXiv:1612.08498 (2016).