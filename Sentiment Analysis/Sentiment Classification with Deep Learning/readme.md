# Sentiment Classification with Deep Learning
  Sentiment analysis or opinion mining refers to the task of classifying texts based
on their affect i.e. whether the expressed opinion is positive, negative or neutral.
Some advanced opinion mining tasks classify texts on the basis of mood (e.g.
happy, angry, sad, etc.), expression (e.g. sarcastic, didactic, philosophical, etc.)
or intention (e.g. question, complaint, request, etc.) For this project, you
will consider a simple binary classification problem that detects whether the
sentiment associated with text is positive or negative.
You will use the IMDB Movie Review Dataset that contains 1000 training
and testing examples. The model is to be built and trained on the training
examples, and to be tested against the test examples.

## Dataset:
  IMDB The IMDB dataset consists of 100,000 movie reviews with binary classes [Maas et al., 2011]. One key aspect of this dataset is that each movie review has several sentences.

  Tools: Numpy, Pandas, NLTK, Word2Vec, Keras, Sk-learn.

## Steps:
  Load the data
  Build the Embedding dictionary
  Create a TensorDataset and DataLoader
  Define the baseline model
  Add an RNN layer to the baseline model
  Replace the RNN by self-attention
  Train and test the model

## Conclusion: 
  Vanilla RNN vs GRU vs LSTM:
  All RNNs have feedback loops in the recurrent layer. This lets them maintain information in 'memory' over time. However, it can be difficult to train standard RNNs to solve problems that require learning long-term temporal dependencies. This is because the gradient of the loss function decays exponentially with time (called the vanishing gradient problem). LSTM networks are a type of RNN that uses special units in addition to standard units. LSTM units include a 'memory cell' that can maintain information in memory for long periods of time. A set of gates is used to control when information enters the memory, when it's output, and when it's forgotten. This architecture lets them learn longer-term dependencies. GRUs are similar to LSTM, but it uses a simplified structure. They also use a set of gates to control the flow of information, but they don't use separate memory cells, and they use fewer gates.
  Adding more layers not always result in more accuracy in neural networks. Adding more layers will help you to extract more features. But we can do that up to a certain extent. There is a limit. After that, instead of extracting features, it will tend to ‘overfit’ the data. Overfitting can lead to errors in some or the other form like false positives.
  According to the information, Bidirectional LSTMs are an extension of traditional LSTMs that can improve model performance on sequence classification problems.
  A neural network armed with an attention mechanism can actually understand what “it” is referring to. That is, it knows how to disregard the noise and focus on what’s relevant, how to connect two related words that in themselves do not carry markers pointing to the other.
  In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations.

## Reference:
  [Chung et al., 2014] Junyoung Chung, Caglar Gulcehre,, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.
  [Socher et al., 2012] Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. Semantic compositionality through recursive matrix-vector spaces. In Proceedings of EMNLP, pages 1201–1211, 2012. 
  [Socher et al., 2013] Richard Socher, Alex Perelygin, Jean Y Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over r a sentiment treebank. In Proceedings of EMNLP, 2013.
  [Collobert et al., 2011] Ronan Collobert, Jason Weston, Leon Bottou, Michael Karlen, Koray Kavukcuoglu, and ´ Pavel Kuksa. Natural language processing (almost) from scratch. The Journal of Machine Learning Research, 12:2493–2537, 2011.
  [Kalchbrenner et al., 2014] Nal Kalchbrenner, Edward network for modelling sentences. In Proceedings of ACL, 2014.
  [Ashish Vaswani, Noam Shazeer et al., 2017] Illia Polosukhin, Attention Is All You Need, arXiv.org > cs > arXiv:1706.03762, 2017

## External links:
  https://blog.csdn.net/omnispace/article/details/100660324
