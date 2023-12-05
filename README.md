# 3D-PCQA
Activating Frequency and ViT for 3D Point Cloud Quality Assessment Without Reference IEEE ICIP 2023 Conference.

This is the implemntation for the paper above.
Link to pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10328373


# Run example for test:

python test.py /path/to/host/model_weights.pt /path/to/host/dataset/ 


# Requirements

einops==0.6.0
numpy==1.23.4
pyntcloud==0.3.1
PyYAML==5.4.1
scikit-learn==1.2.2
pytorch==1.13.1 
pytorch-cuda=11.7
torchvision==0.14.1 
torchaudio==0.13.1 
