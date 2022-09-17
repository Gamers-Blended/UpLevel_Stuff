from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model (binary to model)
model = pickle.load(open('dt_wine.pkl', 'rb')) # rb = read binary
# for keras models, use joblit or model.save

# homepage.com/ -> home
@app.route("/")
# triggers function called index
def index():
    # when triggered, return a string, hello world
    # return "Hello World"
    return render_template('home.html')

# home/hello
@app.route("/hello")
def hello():
    return "Hello hello!"

@app.route("/hello/<string:name>")
def hello_name(name):
    # return "Hello " + name + "!"
    return render_template('hello_name.html', injected_name=name)

@app.route("/predict")
def predict():
    # get values from browser URL
    # request argument, in URL, look for particular string
    # these are the features (col names)
    fixed_acidity = request.args['fixed_acidity']
    volatile_acidity = request.args['volatile_acidity']
    citric_acid = request.args['citric_acid']
    residual_sugar = request.args['residual_sugar']
    chlorides = request.args['chlorides']
    free_sulfur_dioxide = request.args['free_sulfur_dioxide']
    total_sulfur_dioxide = request.args['total_sulfur_dioxide']
    density = request.args['density']
    pH = request.args['pH']
    sulphates = request.args['sulphates']
    alcohol = request.args['alcohol']

    # create 
    # reshape to 1 row, 11 columns
    testData = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1,11)

    # or df
    # testData = pd.DataFrame({"fixed_acidity":[fixed_acidity], })

    class_predicted = model.predict(testData)[0]

    output = "Is the wine tasty? " + str(class_predicted)

    return output

if __name__ == "__main__":
    # debug=True: when changing code, website is dynamically changed
    # else have to kill server and re-run from scratch
    app.run(debug=True)