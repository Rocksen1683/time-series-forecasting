import yfinance as yf
import numpy
import pandas
import datetime
import matplotlib.pyplot
import pickle
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error

stock_name = input('Enter the 4 letter code of your desired stock : ')
stock = yf.Ticker(stock_name)                                                                 # creating an object for the stock

try:                                                                                          # if data for the stock is already present we check
    df = pandas.read_csv("C:\\Users\\rrohi\\OneDrive\\Documents\\{}.csv".format(stock_name))  # if it is updated
    print('File exists. Checking if it is updated...')
    for x in df['Date'][::-1]:                                                                # getting latest date on existing csv    
        last_on_csv = x
        break
    today = datetime.date.today()                                                             # getting today's date 
    hist = stock.history(start=last_on_csv, end=today)
    if len(hist) > 2:                                                                         # to check if there is newer data
        hist = stock.history(period='max')
        hist.to_csv("C:\\Users\\rrohi\\OneDrive\\Documents\\TSLA.csv")                        # updating data
        df = pandas.read_csv("C:\\Users\\rrohi\\OneDrive\\Documents\\TSLA.csv")               # redefining DataFrame object 
        print('Databased updated')
        train_model = True
    else:
        print('Datebase already up to date')
        train_model = False
except:
    hist = stock.history(period = 'max')
    hist.to_csv("C:\\Users\\rrohi\\OneDrive\\Documents\\{}.csv".format(stock_name))
    train_model = True
    
df = pandas.read_csv("C:\\Users\\rrohi\\OneDrive\\Documents\\{}.csv".format(stock_name))      # loading in the data

matplotlib.pyplot.figure()

pandas.plotting.lag_plot(df['Close'], lag=3)                                                  # plot of stock prices plotted with itself in the past with certain lag
matplotlib.pyplot.show()                                                                      # eg June 14 on y axis, June 11 on x axis

matplotlib.pyplot.plot(df["Date"], df["Close"])                                               # plotting price v time
matplotlib.pyplot.xticks(numpy.arange(0,len(df), len(df)//3), df['Date'][0:len(df):len(df)//3])
matplotlib.pyplot.show()

train_data, test_data = df[0:int(len(df)*0.75)], df[int(len(df)*0.75):]                       # 75% of data used for training and 25% for testing
training = train_data['Close'].values                                   
testing = test_data['Close'].values

frame = pandas.DataFrame(training)                                                            # creating a dataframe object
training = frame.diff(periods = 1, axis = 0)                                                  # differencing the training
                                                                                              #periods = 1 for first order differencing and axis = 0 since we're differencing across rows not columns
training = training.to_numpy()

matplotlib.pyplot.plot(training)                                                              # data has been made into a stationary series
matplotlib.pyplot.show()

t = training[1:len(training)+1]

history = [x for x in t]
model_predictions = []
N_test_observations = len(test_data)
progress_count = 0

if train_model:
    print('Training model...')
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = testing[time_point]
        history.append(true_test_value)
        progress_count += 1
        print(progress_count, 'out of', N_test_observations, 'complete.')

    MSE_error = mean_squared_error(testing, model_predictions)          
    print('Testing Mean Squared Error is', MSE_error)    
    print('The predicted value is :', model_predictions[-1][0])

    test_set_range = df[int(len(df)*0.75):].index
    matplotlib.pyplot.plot(test_set_range, model_predictions, color = 'blue', linestyle = 'dashed',label = 'Predicted Price', marker = '.', ms = 4)
    matplotlib.pyplot.plot(test_set_range, testing, color='red', label='Actual Price')
    matplotlib.pyplot.show()

    print('Saving values...')
    file = open("C:\\Users\\rrohi\\OneDrive\\Documents\\{}_values.dat".format(stock_name), 'wb+')
    pickle.dump(model_predictions, file)
    pickle.dump(MSE_error, file)
    file.close()
    print('Saved')
else:
    print('Model already trained.')
    file = open("C:\\Users\\rrohi\\OneDrive\\Documents\\{}_values.dat".format(stock_name), 'rb+')
    try:
        while True:
            x = pickle.load(file)
            if type(x) == type([]):
                print('The predicted value is :', round(x[-1][0], 4))
            else:
                print('Testing Mean Squared Error is', round(x, 4))
    except:
        file.close()
