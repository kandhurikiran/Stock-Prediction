{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [31/Jan/2021 17:03:46] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [31/Jan/2021 17:03:51] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle\n",
    "from werkzeug.serving import run_simple\n",
    "from werkzeug.wrappers import Request, Response\n",
    "import pandas as pd\n",
    "import datetime\n",
    "from datetime import datetime\n",
    "\n",
    "\n",
    "app = Flask(__name__, template_folder='template')\n",
    "\n",
    "model = pickle.load(open('model.pkl', 'rb'))\n",
    "\n",
    "\n",
    "\n",
    "@app.route(\"/\")\n",
    "def home():\n",
    "    return render_template('Trail.html')\n",
    "\n",
    "@app.route(\"/predict\", methods = ['POST'])\n",
    "def predict():\n",
    "    values = [x for x in request.form.values()]\n",
    "    final_features = np.array(values, dtype = 'str')\n",
    "    prediction = model.predict(start = pd.to_datetime(final_features[0]), end = pd.to_datetime(final_features[1]), dynamics = False)\n",
    "    return render_template('Trail.html', prediction_text = prediction)\n",
    "\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
