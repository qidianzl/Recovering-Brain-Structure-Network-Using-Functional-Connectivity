# Recovering-Brain-Structure-Network-Using-Functional-Connectivity
CODE

This is the source code of the following two papers:

[1] Zhang, Lu, Li Wang, and Dajiang Zhu. "Recovering brain structural connectivity from functional connectivity via multi-gcn based generative adversarial network." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.

[2] Zhang, Lu, Li Wang, and Dajiang Zhu. "Predicting Brain Structure Network using Functional Connectivity."  in process.

Paper [1] proposes the Multi-GCN GAN model and structure preserving loss, paper [2] further expands the research on different datasets, different functional connectivity generation methods and different models.

DETAILS:

model.py

We implemented different models here, including two different CNN-based generators, GCN-based generator and GCN-based discriminator. The learnable combination coefficients of multiple GCNs (theta) also can be initialized by different values in this file. We also compared different topology updating methods. We have annotated in this file about how to adopt different methods.

train.py



DATA

We used 1064 subjects from HCP dataset and 132 subjects from ADNI dataset in our research. For each subject we generated the structural connectivity (SC) and the functional connectivity (FC). All of the connectivity matrices can be shared for research purpose. Please contack the author to obtain the data by sending email to lu.zhang2@mavs.uta.edu.


Citation:

If you used the code or data of this project, please cite the two papers:

@inproceedings{zhang2020recovering,
  title={Recovering brain structural connectivity from functional connectivity via multi-gcn based generative adversarial network},
  author={Zhang, Lu and Wang, Li and Zhu, Dajiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={53--61},
  year={2020},
  organization={Springer}
}

