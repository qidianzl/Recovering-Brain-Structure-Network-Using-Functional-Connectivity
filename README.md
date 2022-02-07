# Recovering-Brain-Structure-Network-Using-Functional-Connectivity
### Framework:
![framework](main3.png)

### Papers:
This repository provides a PyTorch implementation of the models adopted in the two papers:

- Zhang, Lu, Li Wang, and Dajiang Zhu. "Recovering brain structural connectivity from functional connectivity via multi-gcn based generative adversarial network." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
- Zhang, Lu, Li Wang, and Dajiang Zhu. "Predicting Brain Structure Network using Functional Connectivity."  in process.

The first paper proposes the Multi-GCN GAN model and structure preserving loss, and the second paper further expands the research on different datasets, different atlases, different functional connectivity generation methods, different models, and new evaluation measures. New results have been obtained.


### Code:
#### dataloader.py
This file includes the preprocessing and normalization operations of the data. All the details have been introduced in the two papers. The only element needs to pay attention to is the empty list, which records the ids of the empty ROIs of specific atlases. For example, there are two brain regions in Destrieux Atlas are empty (Medial_wall for both left and right hemispheres). Therefore the corresponding two rows and columns in the generated SC and FC are zeros. We deleted these rows and columns.

#### model.py
We implemented different models in this file, including two different CNN-based generators, Multi-GCN-based generator and GCN-based discriminator. Different models can be chosen by directly calling the corresponding classes when run the train.py file. Different model architectures are as follows:
- CNN (CNN-based generator, MSE loss and PCC loss)
- Multi-GCN (Multi-GCN-based generator, MSE loss and PCC loss)
- CNN based GAN (CNN-based generator and GCN-based discriminator, SP loss)
- MGCN-GAN (Multi-GCN-based generator and GCN-based discriminator, SP loss)

When adopting the proposed MGCN-GAN architecture, the different topology updating methods and differnet initializations of learnable combination coefficients of multiple GCNs (theta) can be directly changed in this file, and we have annotated in this file about how to change them. For Linear regression model, we directly called the *LinearRegression* from *sklearn.linear_model* package.

#### Loss_custom.py
The proposed SP loss includes three components: GAN loss, MSE loss and PCC loss. In this file, we implemented the PCC loss. For the MSE loss and GAN loss, we directly called the loss functions from torch.nn module in train.py file. By directly editing train.py file, different loss functions can be chosen, including:
- GAN Loss
- MSE+GAN loss
- PCC+GAN loss
- SP loss

#### train.py
You need to run this file to start. All the hyper-parameters can be defined in this file.

Run `python ./train.py -atlas='atlas1' -gpu_id=1`. 

Tested with:
- PyTorch 1.9.0
- Python 3.7.0

### Data:
We used 1064 subjects from HCP dataset and 132 subjects from ADNI dataset in our research. For each subject, we generated the structural connectivity (SC) and the functional connectivity (FC) matrices. All of the connectivity matrices can be shared for research purpose. Please contact the author to get the data by sending email to lu.zhang2@mavs.uta.edu.

### Citation:
If you used the code or data of this project,  please cite:

    @inproceedings{zhang2020recovering,
      title={Recovering brain structural connectivity from functional connectivity via multi-gcn based generative adversarial network},
      author={Zhang, Lu and Wang, Li and Zhu, Dajiang},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={53--61},
      year={2020},
      organization={Springer}
    }


