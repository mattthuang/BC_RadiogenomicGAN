# Conditional generative adversarial network driven radiomic prediction of mutation status based on magnetic resonance imaging of breast cancer

## Introduction 

This repository provides the source code and raw datasets associated with the paper Conditional Generative adversarial network driven radiomic prediction of mutation status based on magnetic resonance imaging of breast cancer. 

Breast Cancer (BC) is a highly heterogeneous and complex disease. Personalized treatment options require the integration of multi-omic data and consideration of phenotypic variability. Radiogenomics aims to merge medical images with genomic measurements but encounter challenges due to unpaired data consisting of imaging, genomic, or clinical outcome data.

We proposed the utilization of a well-trained conditional generative adversarial network (cGAN) to address the unpaired data issue in radiogenomic analysis of BC. The generated images will then be used to predict the mutation status of key driver genes. The overall project workflow is depicted below. 



![Overall study design](/png/overall.png)



## Conditional GAN 

### Architecture 

![cGAN Architecture](/png/cgan_arch.png)



### Data 

The cGAN is trained with matched patient MRI and BTF features obtained from patient multi-omic data. MRIs are in the `matched_sideview` and `matched_sideview_test` directories. Matched BTF features are stored in the `BTF_side.csv`. Unpaired BTF features are stored in `Mu_paired_test.csv`, which represents patients with no imaging data available and will be used for MRI prediction after the cGAN model is tuned and well-trained. 

* raw multi-omics data is obtained from the [TCGA](https://www.cancer.gov/tcga )
* MRI obtained from the [TCIA](https://www.cancerimagingarchive.net), where the digital image pixel values are extracted

### Training 

To train the cGAN model, run: 

```
python main.py
```

Model parameters can be adjusted in `params.py`, where: 

* `epochs` specifies the number of epochs the model will be trained for 
* `batch_size` indicates the batch_size of the model 
* `z_dim` is the dimension of the random noise vector Z 
* `z_dis` is the distribution of the random noise vector
* `g_lr` sets the learning rate for the generator network 
* `d_lr` sets the learning rate for the discriminator network 
* `data_dir` sets the directory for the matched patient MRIs
* `BTF_dir` sets the .csv file for the BTF features for the patients 

### Testing 

To run validation / predictions using the trained cGAN model, run: 

```
python main.py --test true 
```

Relevant parameters are stated in `params.py`, where: 

* `test_dir` sets the directory for patient BTF features. cGAN model will use these features and generate predicted MRIs
* `output_dir` specifies where the generated MRIs will be stored 

### Evaluation 

cGAN evaluation is carried out using the Fréchet Inception Distance (FID) metric, where the distance between the distribution of real and fake images are calculated. The 3D component of the FID relies on the pretrained 3D MedicalNet by [Chen et al](https://github.com/Tencent/MedicalNet), while implementation of the metric was based on the study [Evaluation of 3D GANs for Lung tissue Modelling in Pulmonary CT](https://github.com/S-Ellis/healthy-lungCT-GANs/tree/main) by Ellis et al., 2022. 

To calculate the FID of the generated images, run: 

```
python fid3d.py --real <real_path> --fake <fake_path>
```

* <real_path> is the path to the real patient MRIs
* <fake_path> is the path to the cGAN generated MRIs
* Patient ID must match between the two folders 

##### Example use: 

```
python fid3d.py --real ../testreal/ --fake ../testfake/
```



## Convolutional Neural Network

### Architecture

![CNN Architecture](/png/cnn.png)

### Data

#### Training/Testing dataset construction

To train the CNN model, first identify the preferred proportion for the test set, then run `training_testing.py` to split the mutation status data into a trianing and testing set. 

```
python training_testing.py --file <csv directory> --testpercent <percent> --trainname <training csv filename> --testname <testing csv filename>
```

* `<csv directory>` is the path to the directory of the mutation status csv
* `<percent>` is the desired test proportion as a decimal value 
* `<trainname>` is the desired filename for the constructed training csv 
* `<testname>` is the desired filename for the constructed testing csv

##### Example use: 

```
python training_testing.py --file ./mutation/MutationStatus_all --testpercent 0.1 --trainname train_10percent.csv --testname test_10percent
```

#### Matching MRIs

The next step is to set up a directory containing the corresponding MRIs to each of the patient in the previously constructed datasets by running `matchimage.py`.

```
python matchimage.py --csvfile <file> --imagedir <image_directory> --outputdir <output_directory>
```

* `<file>` represents the path to the dataset csv file of interest 
* `<image_directory>` referes to the directory which contains the cGAN/real patient MRIs
* `<output_directory>` specifies the directory to move MRIs to

##### Example use: 

```
python matchimage.py --csvfile ./mutation/MutationStatus_Train --imagedir ../outputs/dcgan/first_test/test_outputs/ --outputdir ./training_image
```

#### Training & Evaluation 

Performance evaluation of the CNN is based on the AUC value of the ROC and PR curve. Run `CNN.py` and evaluation will proceed after training is completed. 

```
CNN.py --train-data <training_csv> --test-data <testing_csv> --train_image <training_img_dir> --test_image <testing_img_dir> --gene <gene> --epochs <epochs> --lr <learning_rate> --weight-path <weight_path>
```

* `<training_csv>` describes the path of the training csv
* `<testing_csv>` describes the path of the testing csv
* `<training_img_dir>` specifies the directory of the training MRIs
* `<testing_img_dir>` specifies the directory of the testing MRIs
* `<gene>` states the gene of interest that the CNN will be trained for 
* `<epochs>` states the number of epochs that training will partake in 
* `<learning_rate>` specifies the learning rate for the CNN 
* `<weight_path>` specifies the path where the weight of the model will be saved after training has completed

##### Example use: 

```
python final.py --train-data  ./mutation/MutationStatus_Train.csv --test-data ./mutation/MutationStatus_Test.csv --train_image ./training_image_cgan --test_image ./testing_image_cgan --gene TP53 --epochs 2000
```

### Results

* cGAN generated images split and stored in `CNN/testing_image_cgan` and `CNN/training_image_cgan`, where only 46/546 training images and 20/144 testing images are shown due to size restrictions
* Well-train model is provided the `results` directory and training MRIs every 100 epochs are provided due to size restrictions
