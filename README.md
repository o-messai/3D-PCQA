# Activating Frequency and ViT for 3D Point Cloud Quality Assessment Without Reference. IEEE ICIP 2023 Conference. </br>
## 3D-PCQA </br>
This is the implemntation for the paper above.
Link to pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10328373

![3D_mesh](https://github.com/o-messai/3D-PCQA/assets/10109223/3247790c-5a52-42dc-b6f5-8906d8b65d12)


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

            @proceeding{messai2023-mesh3d,
            title={Activating Frequency and VIT for 3D Point Cloud Quality Assessment Without Reference},
            author={Messai, Oussama and Bentamou, Abdelouahid and Zein-Eddine, Abbass and Gavet, Yann},
            conference={ICIPCW2023},
            year={2023},
            publisher={IEEE},
            doi={10.1109/ICIPC59416.2023.10328373}}




