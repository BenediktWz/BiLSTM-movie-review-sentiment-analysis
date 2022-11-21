Project Sentiment Analysis - Classification of Movie Reviews

## Description:
The task was to build neural network-based models to predict the sentiment of the reviews from Roten Tomatoes’s [https://www.rottentomatoes.com/]. The target variable consists of the five sentiment categories:
- 0: Negative
- 1: Somewhat Negative
- 2: Neutral
- 3: Somewhat Positive
- 4: Positive
This task is based on a Kaggle competition [https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews].
Our final model was a bidirectional LSTM neural network with BERT embeddings. The GloVe emebddings resulted in a very similar accuracy as the BERT embeddings. We used the pytorch library to build the BiLSTM model and the transformers library for the BERT embeddings.

## Project Structure:
- `Models-Archive`: Here we present our early approaches to the Problem with different Model Architectures. The Models follow the same core structure besides the shallow learning approach Logistic Regression. These Notebooks were our testing environments.
- `Train_GloVe_BiLSTM_Final` This Notebook is the result of our early approaches. It was used to tune the Hyperparameters.
- `Train_BERT_BiLSTM_Final` This Notebook is used as comparison, how the use of the BERT Transformer and Transfer Learning could improve the results. The pretrained-BERT Model was used and a final LSTM Layer was then trained on the data set.
- `Evaluation_GloVe_BiLSTM_Final` This is the final evaluation on the Test-Set, the model here is built with the findings of the Finetuning.
- `Train_GloVe_BiLSTM_kFold_CV` is a notebook that is an adaption from GloVE-BiLSTM that uses Cross Validation to verify the results. We did decide against using Cross Validation for every Notebook as it poses a big increase in runtime.

## Provided Data

- `train.tsv` We use the same training data as provided by the kaggle competition. 
- `test_mapped.tsv` Test_mapped.tsv was created by mapping test.tsv based on the Phrase on dictionary.txt and then mapping dictionary.txt based on the PhraseID on sentiment_labels.txt. Which in return delivered us the labels for those phrases in test.tsv which are also included in the dictionary.txt.

## Important Notices
* This project uses pre-trained word embeddings. Downloading them the first time takes a few minutes.

* Also the spacy tokenizer is used and can be installed via:

```
pip install spacy
python -m spacy download en_core_web_sm
```
