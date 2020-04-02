# Long and Diverse Text Generation with Planning-based Hierarchical Variational Model

## Introduction

Existing neural methods for data-to-text generation are still struggling to produce long and diverse texts: they are insufficient to model input data dynamically during generation, to capture inter-sentence coherence, or to generate diversified expressions. To address these issues, we propose a Planning-based Hierarchical Variational Model (PHVM). Our model first plans a sequence of groups (each group is a subset of input items to be covered by a sentence) and then realizes each sentence conditioned on the planning result and the previously generated context, thereby decomposing long text generation into dependent sentence generation sub-tasks. To capture expression diversity, we devise a hierarchical latent structure where a global planning latent variable models the diversity of reasonable planning and a sequence of local latent variables controls sentence realization. 

This project is a Tensorflow implementation of our work.

## Requirements

*   Python 3.6
*   Numpy
*   Tensorflow 1.4.0

## Quick Start

*   Dataset
  
    Our dataset contains 119K pairs of product specifications and the corresponding advertising text. For more information, please refer to our paper.
    
*   Preprocess

    *   Download data from https://drive.google.com/open?id=1vB0fT1ex2Tsid-i5s-jqdz9QUFbCh0CO and unzip the file, which will create a new directory named `data`. The path to our dataset is `./data/data.jsonl`.
    *   We provided most preprocessed data under `./data/processed/` except pre-trained word embeddings which can be generated with the following command line:
    
    ```
    bash preprocess.sh
    ```

*   Train

    ```
    ./run.sh
    ```

*   Test

    ```
    ./test.sh
    ```

## Citation

Our paper is available at https://arxiv.org/abs/1908.06605v2.

**Please kindly cite our paper if this paper and the code are helpful.**
