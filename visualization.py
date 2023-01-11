from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pickle
import numpy as np

def read_pickle(file_path) :
    file_to_read = open(file_path, "rb")
    data_read = pickle.load(file_to_read)
    return data_read

def plot_wordcloud(text, mask=None, max_words=None, max_font_size=100, figure_size=(24.0, 16.0),
                   title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black',
                                   'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()
    if(title == "Spam Email"):
        plt.savefig("spam.png")
    else:
        plt.savefig("nonSpam.png")


if __name__ == '__main__':
    # Plotting Word Cloud
    X_train = np.array(read_pickle("Train_X.pkl"), dtype=object)
    Y_train = np.array(read_pickle("Train_Y.pkl"))
    X_test = np.array(read_pickle("Test_X.pkl"), dtype=object)
    Y_test = np.array(read_pickle("Test_Y.pkl"))
    '''for i, o in enumerate(X_train):
        if o is None:
            continue
        print("-----" + str(i)+ "------")
        print(o)
        print("\n") '''
    spam_train_index = [i for i, o in enumerate(Y_train) if o == 1]
    non_spam_train_index = [i for i, o in enumerate(Y_train) if o == 0]
    spam_email = np.array(X_train)[spam_train_index]
    non_spam_email = np.array(X_train)[non_spam_train_index]
    plot_wordcloud(spam_email, title='Spam Email', max_words=200)
    plot_wordcloud(non_spam_email, title="Non Spam Email", max_words=200)

    # N-grams model visualization

