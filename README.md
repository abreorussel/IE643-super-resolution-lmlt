# Generating High Resolution Zoom-IN for Images using LMLT-Base-x2 (Low-to-high Multi-Level Vision Transformer)

Russel Abreo ,Anand Patel


### Requirements
```
# Install Packages
pip install -r requirements.txt
pip install matplotlib

# Install BasicSR
python3 setup.py develop
```


### Dataset
The model is trained on DIV2K, Flickr2K.
You can download two datasets at https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
and prepare other test datasets at https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Common-Image-SR-Datasets.

The model is finetuned on RealSR dataset. The link for the dataset is https://www.kaggle.com/datasets/yashchoudhary/realsr-v1.

### Preprocessing the dataset

Creating a compatible directory structure.
```
python process_real_sr_dataset.py --directory_path "path_to_dataset_directory" --new_directory_name "realsr"  --scale "2"
```
And also, you'd better extract subimages using 
```
python3 scripts/data_preparation/extract_subimages_realsr.py
```

By running the code above, you will get subimages of RealSR dataset.


### Finetune
You can finetune LMLT with the following command below.
```
python3 basicsr/train.py -opt options/finetune/LMLT/finetune_base_RealSR_X2.yml
```


### Test
You can test LMLT following commands below
```
python3 basicsr/test.py -opt options/test/LMLT/test_base_benchmark_X2.yml
```

### Refer the lmlt_notebook.ipynb for finetuning steps.
### Refer the IE643_Final_Streamlit_Interface_Crop.ipynb for the Interface.
### The interface code is mentionend in streamlit.py



## Credits

This project builds upon the work of others:

- **Research Paper**:  
  [*Title of the Paper*](https://www.arxiv.org/abs/2409.03516) by Jeongsoo Kim, Jongho Nang, Junsuk Choe<sup>*</sup>. Published in 2024*.

- **Code Repository**:  
  Original implementation by [GitHubUser](https://github.com/jwgdmkj/LMLT/tree/main) on GitHub.

