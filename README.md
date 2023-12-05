# Activating Frequency and ViT for 3D Point Cloud Quality Assessment Without Reference IEEE ICIP 2023 Conference. </br>
## 3D-PCQA </br>

This is the implemntation for the paper above.
Link to pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10328373


## Run example for test:

python test.py /path/to/host/model_weights.pt /path/to/host/dataset/ 

## Requirements

einops==0.6.0  </br>
numpy==1.23.4  </br>
pyntcloud==0.3.1  </br>
PyYAML==5.4.1 </br>
scikit-learn==1.2.2 </br>
pytorch==1.13.1  </br>
pytorch-cuda=11.7 </br>
torchvision==0.14.1  </br>
torchaudio==0.13.1  </br>

## Citation

  @INPROCEEDINGS{10328373,
    author={Messai, Oussama and Bentamou, Abdelouahid and Zein-Eddine, Abbass and Gavet, Yann},
    booktitle={2023 IEEE International Conference on Image Processing Challenges and Workshops (ICIPCW)}, 
    title={Activating Frequency and VIT for 3D Point Cloud Quality Assessment Without Reference}, 
    year={2023},
    volume={},
    number={},
    pages={3636-3640},
    doi={10.1109/ICIPC59416.2023.10328373}}



