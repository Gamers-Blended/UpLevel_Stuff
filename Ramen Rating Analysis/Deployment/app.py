import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('logr_model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

# Provide features to model
@app.route('/predict', methods=['POST'])
def predict():
	'''
	For rendering results on HTML GUI
	'''

	'''
	Initialize all variables with default value, 0
	'''
	# Flavours
	isSpicy = 0
	hasChicken = 0
	hasBeef = 0
	hasSeafood = 0

	if request.form.get("isSpicy"):
		isSpicy = request.form.getlist('isSpicy')[0]
	if request.form.get("hasChicken"):
		hasChicken = request.form.getlist('hasChicken')[0]
	if request.form.get("hasBeef"):
		hasBeef = request.form.getlist('hasBeef')[0]
	if request.form.get("hasSeafood"):
		hasSeafood = request.form.getlist('hasSeafood')[0]

	flavour_features = [isSpicy, hasChicken, hasBeef, hasSeafood]
	flavour_features = [int(x) for x in flavour_features]

	# Brand
	brand_features = [0] * 30
	brand_dict = {
		"0": "Acecook",
		"1": "Indomie",
		"2": "Itsuki",
		"3": "JML",
		"4": "KOKA",
		"5": "Lucky Me!",
		"6": "MAMA",
		"7": "Maggi",
		"8": "Mama",
		"9": "Mamee",
		"10": "Maruchan",
		"11": "Master Kong",
		"12": "MyKuali",
		"13": "Myojo",
		"14": "Nissin",
		"15": "Nongshim",
		"16": "Other",
		"17": "Ottogi",
		"18": "Paldo",
		"19": "Samyang Foods",
		"20": "Sapporo Ichiban",
		"21": "Sau Tao",
		"22": "Ve Wong",
		"23": "Vedan",
		"24": "Vifon",
		"25": "Vina Acecook",
		"26": "Wai Wai",
		"27": "Wei Lih",
		"28": "Wu Mu",
		"29": "Yum Yum"
	}

	if request.form.get("brand"):
		selected_brand = request.form.getlist('brand')[0]

		for key, value in brand_dict.items():
			if selected_brand in value:
				brand_features[int(key)] = 1
	print(selected_brand)
	print(brand_features)

	# Serving Type
	serving_features = [0] * 4
	serving_dict = {
		"0": "Cup",
		"1": "Other",
		"2": "Pack",
		"3": "Tray"
	}

	if request.form.get("served"):
		selected_serving = request.form.getlist('served')[0]

		for key, value in serving_dict.items():
			if selected_serving in value:
				serving_features[int(key)] = 1
	print(selected_serving)
	print(serving_features)

	# Country of Origin
	country_features = [0] * 10
	country_dict = {
		"0": "Hong Kong",
		"1": "Indonesia",
		"2": "Japan",
		"3": "Malaysia",
		"4": "Other",
		"5": "Singapore",
		"6": "South Korea",
		"7": "Taiwan",
		"8": "Thailand",
		"9": "United States"
	}

	if request.form.get("country"):
		selected_country = request.form.getlist('country')[0]

		for key, value in country_dict.items():
			if selected_country in value:
				country_features[int(key)] = 1
	print(selected_country)
	print(country_features)

	'''
	Model selection
	'''
	if request.form.get("model"):
		selected_model = request.form.getlist('model')[0]

		if selected_model == 'logr':
			model = pickle.load(open('logr_model.pkl', 'rb'))
		elif selected_model == 'dtclf':
			model = pickle.load(open('dtclf_model.pkl', 'rb'))
		elif selected_model == 'rfclf':
			model = pickle.load(open('rfclf_model.pkl', 'rb'))
		else:
			pass

	# Takes the input from entire form
	int_features = flavour_features + brand_features + serving_features + country_features
	print(int_features)

	# Convert to np array
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)

	# Nearest 2 dp
	# output = round(prediction[0], 2)
	if prediction[0] == 0:
		output = "Bad"
	else:
		output = "Good"

	# Collect inputs
	selected_flavour = []

	if isSpicy == '1':
		selected_flavour.append("Spicy")
	if hasChicken == '1':
		selected_flavour.append("Chicken")
	if hasBeef == '1':
		selected_flavour.append("Beef")
	if hasSeafood == '1':
		selected_flavour.append("Seafood")

	# prediction_text will be replaced by index.html
	return render_template('index.html', prediction_text='Predicted Tier is {}'.format(output),
	 selected_flavour=selected_flavour,
	 selected_brand=selected_brand,
	 selected_serving=selected_serving,
	 selected_country=selected_country)


# Runs Flask
if __name__ == "__main__":
	app.run(debug=True)