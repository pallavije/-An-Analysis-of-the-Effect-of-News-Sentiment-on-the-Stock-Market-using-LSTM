# Stock open/close value prediction using LSTM based on news headline

Dataset : [click](https://www.kaggle.com/BidecInnovations/stock-price-and-news-realted-to-it)
          [click](https://fred.stlouisfed.org/series/NASDAQCOM)

## Process:
- Embedding matrix of headlines based glove
- Min-Max normalization of open/close values
- LSTM sequential model for developing values prediction model
- Deployement of model using Flask


## Libraries:
- tensorflow
- Flask
- matplotlib
- nltk
- numpy
- wordcloud
- seaborn
- Keras
- pandas
- scikit_learn

## Usage:
1) Install all the libraries using "pip install -r requirements.txt"
2) To train, run the StockPredictions.py
3) Open cmd in project folder
4) Enter python webapp.py
5) Open browser and Predict !


