# Interpretable image classification through an argumentative dialog between encoders
This is the code of the paper '[Interpretable image classification through an argumentative dialog between encoders](https://ebooks.iospress.nl/doi/10.3233/FAIA240880)' published at ECAI 2024. 

## Settings

Download the 'last_100.pt' file [here](https://zenodo.org/record/8124014) and place it in the 'pretrained_model' directory.
You will also need to download the CUB200 and/or the Flowers120 dataset.

The dependencies are listed in the 'requirements.txt' file and can be installed using the pip command below:

```
pip3 install -r requirements.txt
```

## Experiments
Here are the 4 scripts corresponding to the 4 DEBATES experiments on CUB with DINO and DINOv2 and on Flowers with DINO and DINOv2.

```
python3 main.py --config-file configs/config_CUB_DINO.yaml --data-path data/path/CUB
python3 main.py --config-file configs/config_CUB_DINOv2.yaml --data-path data/path/CUB
python3 main.py --config-file configs/config_flowers_DINO.yaml --data-path data/path/Flowers
python3 main.py --config-file configs/config_flowers_DINOv2.yaml --data-path data/path/Flowers
```

where data/path/CUB and data/path/Flowers is replaced by the path to the dataset (CUB or Flowers).

We also give a way to save dialogues in a textual and graph form as presented in the supplementary. 
This is done with the argument --visualize, it can take the value 'all' to save all dialogues or 'error' to save only misclassifications


Each dialogue file is identified with a couple (epoch, batch position).
