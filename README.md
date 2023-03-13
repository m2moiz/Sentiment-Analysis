# Realtime Sentiment Analysis

This is a real-time sentiment analysis project that uses machine learning to analyze the sentiment of tweets as they are posted on Twitter. The model is trained using the [Sentiment140 dataset](http://help.sentiment140.com/for-students/)  and uses a combination of natural language processing (NLP) techniques and a support vector machine (SVM) classifier to predict the sentiment of tweets.
## Requirements
- Python 3.7 or higher
- NumPy
- Pandas
- Scikit-learn
- Tweepy
- TextBlob

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```


## Running the Application 
1. Create a Twitter account if you don't already have one. 
2. Apply for a [developer account](https://developer.twitter.com/en/apply-for-access)  to get access to Twitter's API. 
3. Once your developer account is approved, create a new Twitter app and generate your API keys and access tokens. 
4. Open the `config.py` file and enter your API keys and access tokens. 
5. Open a terminal and navigate to the root directory of the project. 
6. Run the following command to start the application:

```bash
python app.py
``` 
7. The application will start streaming tweets in real-time and analyzing their sentiment. You can view the results in the console.
## Customization

You can customize the application by modifying the following files: 
- `config.py`: This file contains the configuration settings for the application, such as the Twitter API keys and access tokens. 
- `app.py`: This file contains the code for streaming tweets and analyzing their sentiment. You can modify the code to change the criteria for selecting tweets or to use a different machine learning model.
## Want to Contribute?

If you find a bug or would like to suggest a new feature, please feel free to open an issue or submit a pull request.
## License

This project is licensed under the MIT License
