# kontex

A machine learning summarizer inspired by the conference paper "Automatic Text Summarization Using a Machine Learning Approach" in the [16th Brazilian Symposium on Artificial Intelligence](https://link.springer.com/book/10.1007/3-540-36127-8) Conference in 2002.

The purpose of the model is to grab extractive summaries of a document by extracting sentences, categorized by:
  1. Mean-TF-ISF
  2. Sentence Length
  3. Sentence Position
  4. Similarity to Title
  5. Similarity to Keywords
  6. Sentence-to-Sentence Cohesion
  7. Sentence-to-Centroid Cohesion
  8. Depth in the tree (related to Sentence-to-Centroid Cohesion)
  9. Referring position in a given level of the tree (positions 1, 2, 3, and 4) (related to Sentence-to-Centroid Cohesion)
  10. Indicator of main concepts
  11. Occurrence of proper names
  12. Occurrence of anaphors
  13. Occurrence of non-essential information
  
We then run this model against a normal extractive summarizer with the [20 news groups dataset](http://qwone.com/~jason/20Newsgroups/) to train it to grab extractive summaries.

This project was made for "human learning of machine learning" - Jeffrey Fei
