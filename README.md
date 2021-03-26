# Malware Classification using classical Machine Learning and Deep Learning

This repository is the official implementation of the research mentioned in the chapter **"An Empirical Analysis of Image-Based Learning Techniques for Malware Classification"** of the Book **"Malware Analysis Using Artificial Intelligence and Deep Learning"**

The book or chapters can be purchased from: https://link.springer.com/chapter/10.1007/978-3-030-62582-5_16

The arXiv eprint is at: https://arxiv.org/abs/2103.13827

![alt text](https://media.springernature.com/w306/springer-static/cover-hires/book/978-3-030-62582-5)


### Abstract

In this chapter, we consider malware classification using deep learning techniques and image-based features. We employ a wide variety of deep learning techniques, including multilayer perceptrons (MLP), convolutional neural networks (CNN), long short-term memory (LSTM), and gated recurrent units (GRU). Among our CNN experiments, transfer learning plays a prominent role—specifically, we test the VGG-19 and ResNet152 models. As compared to previous work, the results presented in this chapter are based on a larger and more diverse malware dataset, we consider a wider array of features, and we experiment with a much greater variety of learning techniques. Consequently, our results are the most comprehensive and complete that have yet been published.

### Quick Notes:

* Classic ML-based approaches tried : K-NN, Random Forest, and XGBoost
* Deep Learning-based approaches tried: ANN, CNN, LSTM, and GRU
* Implementation is using sklearn, numpy, pandas and pytorch.
* MS Windows executable binary files are used as data.
* Features
  * Classic ML-based approaches: PE fie features are extraced and used
  * Deep Learning-based approaches: (1) Opcodes (2) Converted executables into gray-scale images
* This project is an extension of https://github.com/pratikpv/malware_classification


### If you like our work and is useful for your research please cite this chapter/paper as:
```
Prajapati P., Stamp M. (2021) An Empirical Analysis of Image-Based Learning Techniques for Malware Classification. In: Stamp M., Alazab M., Shalaginov A. (eds) Malware Analysis Using Artificial Intelligence and Deep Learning. Springer, Cham. https://doi.org/10.1007/978-3-030-62582-5_16
```
or
```
@Inbook{
    Prajapati2021,
    author={Prajapati, Pratikkumar and Stamp, Mark},
    editor={Stamp, Mark and Alazab, Mamoun  and Shalaginov, Andrii},
    title={An Empirical Analysis of Image-Based Learning Techniques for Malware Classification},
    bookTitle={Malware Analysis Using Artificial Intelligence and Deep Learning},
    year={2021},
    publisher={Springer International Publishing},
    address={Cham},
    pages={411-435},
    abstract={In this chapter, we consider malware classification using deep learning techniques and image-based features. We employ a wide variety of deep learning techniques, including multilayer perceptrons (MLP), convolutional neural networks (CNN), long short-term memory (LSTM), and gated recurrent units (GRU). Among our CNN experiments, transfer learning plays a prominent role---specifically, we test the VGG-19 and ResNet152 models. As compared to previous work, the results presented in this chapter are based on a larger and more diverse malware dataset, we consider a wider array of features, and we experiment with a much greater variety of learning techniques. Consequently, our results are the most comprehensive and complete that have yet been published.},
    isbn={978-3-030-62582-5},
    doi={10.1007/978-3-030-62582-5_16},
    url={https://doi.org/10.1007/978-3-030-62582-5_16}
}
```

