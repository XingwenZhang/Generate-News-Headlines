# Generate Chinese News Headlines with LSTM

## Goal:

We want to take use of LSTM to generate the headlines of news in Chinese, which could be used in search engine with brief description for the searched results.

## Requirements

```
Python >= 3.5
TensorFlow >= 1.0
Keras >= 2.0.2
numpy >= 1.12.1
```

## Usage

```
The application of our finding could be used in text summarization. Which could be used in the field like auto-generate summary of text in the news app or websites. Editors have no need to think about the headline by themselves. What’s more, we think it evenly could be used in the search engine. We all know like Google, under each search result, there is a snippet which show the sentence which contains the query keywords. If this skill is mature, it could help search engine to build up more efficient snippet, which could tell users more information.
```

## How it Works

```
For our experiment, we take Chinese news with headlines as our source data, which is quite different from the English. Chinese cannot be split only according to the blank space; all Chinese characters are connected with each other (except the punctuations). Chinese words can be split by their meanings. For example, two or three or more Chinese characters could be regarded as one meaning group. Therefore, for processing the data, we take use of the Python library called Jieba for Chinese text segmentation. According to the introduction of this library, it is based on trie (prefix tree). It will construct a DAG (directed acyclic graph) that all possible groups of characters. Then it uses dynamic programming to find the path which has highest probability. For the unknown words, it uses HMM model with Viterbi algorithm. 
After getting the words group. We take use of the word2Vec to build up word embedding, which could solve the problem of sparse representation of word vector and it builds up efficient word vector representation for usage. We make use of Google word2vec application for our Chinese news. Then we put the word vector as input of our LSTM. In the neural network model, we build up 4 hidden layers with attention mechanism. We also take use of Dropout (randomly drop out some proportion of data) to avoid overfitting. For training the LSTM, we also use learning rate decay method, which is a good balance between training speed and training accuracy. 
Attention is a mechanism that helps the network remember certain aspects of the input better, the attention mechanism is used when outputting each word in the last layer. We divide the last unit into two parts, one part for computing the attention weight, another part for computing the context. 
```

## Reference

```
Rush, Alexander M., Sumit Chopra, and Jason Weston. "A neural attention model for abstractive sentence summarization." arXiv preprint arXiv:1509.00685 (2015).
Lopyrev, Konstantin. "Generating news headlines with recurrent neural networks." arXiv preprint arXiv:1512.01712 (2015).
Nallapati, Ramesh, et al. "Abstractive text summarization using sequence-to-sequence rnns and beyond." arXiv preprint arXiv:1602.06023 (2016).
Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
```

## Credits

```
Xingwen Zhang, xingwenz@usc.edu
Ruibo He, ruibohe@usc.edu
Yang Cao, cao522@usc.edu
Jiayue Yang, jiayue@usc.edu
```

## License

```
TODO
```

## Contact Us

[LinkedIn](https://www.linkedin.com/in/xingwen-zhang/) 
