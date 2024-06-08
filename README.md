# Activating Frequency and ViT for 3D Point Cloud Quality Assessment Without Reference. IEEE ICIP 2023 Conference. </br>
## 3D-PCQA </br>
This is the implemntation for the paper above.
Link to pdf: 
https://arxiv.org/abs/2312.05972

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10328373

![Screenshot from 2023-12-05 13-02-27](https://github.com/o-messai/3D-PCQA/assets/10109223/152901d9-1c8d-4ec7-a27b-08c79d4947dc)


## Run example for test:

python test.py /path/to/host/model_weights.pt /path/to/host/dataset/ 

## Requirements

- einops  </br>
- numpy  </br>
- pyntcloud  </br>
- pyYAML </br>
- scikit-learn </br>
- pytorch  </br>


## Citation

            @proceeding{messai2023-mesh3d,
            title={Activating Frequency and VIT for 3D Point Cloud Quality Assessment Without Reference},
            author={Messai, Oussama and Bentamou, Abdelouahid and Zein-Eddine, Abbass and Gavet, Yann},
            conference={ICIPCW2023},
            year={2023},
            publisher={IEEE},
            doi={10.1109/ICIPC59416.2023.10328373}}




