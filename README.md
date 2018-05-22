# Word Translation Without Parallel Data and Improvements


Word Translation Without Parallel Data [https://arxiv.org/pdf/1710.04087.pdf]

Author’s code - https://github.com/facebookresearch/MUSE

Monolingual Word Embeddings:

Downloaded FastText Embeddings (Text) from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md


The code includes :
1. Adversarial training to learn a rotation matrix
2. Refinement using Procrustes
3. CSLS distance metric
4. Evaluation using Word Translation, Cross Lingual Semantic Word Similarity and Sentence translation retrieval (Note: I had referred to the author’s code for the evaluation setup)

Code @ https://github.com/abhi252/CSE293_NLP/

To run:
1. Get word embeddings from fast text for two languages
  a. English -> wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
  b. Italian -> wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.vec
2. Run the script to get scores for all tasks for the “Adversarial - Refinement - CSLS” unsupervised technique (after you clone the repository)
  python baseline.py -s <Path to source language .vec file> -t <Path to target language>
     
 
Results:
The results I obtained using my code were comparable to the values in the paper (with minor fluctuations)

References:
1. Pytorch docs
2. Conneau, Alexis, et al. "Word translation without parallel data." arXiv preprint arXiv:1710.04087 (2017).
3. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
