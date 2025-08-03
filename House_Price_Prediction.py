
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "#Load data from CSV file\n",
    "data = pd.read_csv('home_dataset.csv')\n",
    "\n",
    "#Extract features and target variable\n",
    "house_sizes = data ['HouseSize'].values\n",
    "house_prices = data['HousePrice'].values\n",
    "\n",
    "#Visualize the data\n",
    "plt.scatter(house_sizes, house_prices, marker = 'o', color = 'blue')\n",
    "plt.title('House Prices vs. House Size')\n",
    "plt.xlabel('House Size (sq.ft)')\n",
    "plt.ylabel('House Price ($)')\n",
    "plt.show()\n",
    "\n",
    "#Split the data into training and testing sets\n",
    "x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size = 0.2, random_state = 42)\n",
    "\n",
    "#Reshape for Numpy\n",
    "x_train = x_train.reshape(-1, 1)\n",
    "x_test = x_test.reshape(-1, 1)\n",
    "\n",
    "#Create and train the model\n",
    "model = LinearRegression()\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "#Predict prices for the test set\n",
    "Predictions = model.predict(x_test)
    "#Visualize the predictions
    "plt.scatter(x_test, y_test, marker = 'o', color = 'blue', label = 'Actual Prices')
    "plt.plot(x_test, Predictions, color = 'red', linewidth = 2, label = 'Predicted Prices')
    "plt.title('Dumbo Property Price Prediction with Linear Regression')
    "plt.xlabel('House Size (sq.ft)')
    "plt.ylabel('House Price (millions in $)')
    "plt.legend()
    "plt.show()
