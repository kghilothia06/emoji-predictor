# emoji-predictor
a flask-based app for emoji prediction

The training data, 'train_csv.csv' consists of several tweets and the corresponding emoji label

'test_emoji.csv' is for testing model's performance

'Mapping.csv' maps each emoji label to its emoji 

EDA done!

function to create word embeddings for each word in each sentence for training and test data has been coded. It uses pretrained GLOVE vector embeddings for the same.

stacked LSTM model has been built. It achieves 60% accuracy on test set.
