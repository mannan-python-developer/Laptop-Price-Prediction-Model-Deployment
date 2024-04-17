from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('LaptopPricePrediction_With DTR (copy).pkl', 'rb'))
df = pd.read_csv('Saved_Laptop_data.csv')

def fetch_products(selected_company):
    products = sorted(df[df['Company'] == selected_company]['Product'].unique().tolist())
    typeNames = sorted(df[df['Company'] == selected_company]['TypeName'].unique().tolist())
    opSys = sorted(df[df['Company'] == selected_company]['OpSys'].unique().tolist())
    Cpu = sorted(df[df['Company'] == selected_company]['Cpu'].unique().tolist())
    Gpu = sorted(df[df['Company'] == selected_company]['Gpu'].unique().tolist())
    return {'products': products, 'typeNames': typeNames, 'opSys': opSys, 'Cpu': Cpu, 'Gpu': Gpu}

@app.route('/', methods=['GET', 'POST'])
def index():
    Company = sorted(df['Company'].unique())
    Inches = sorted(df['Inches'].unique())
    ScreenResolution = sorted(df['ScreenResolution'].unique())
    Ram = sorted(df['Ram'].unique())
    Weight = sorted(df['Weight'].unique())
    Memory_Size = sorted(df['Memory_Size'].unique())
    Memory_Type = sorted(df['Memory_Type'].unique())

    if request.method == 'POST':
        selected_company = request.form.get('company')
        data = fetch_products(selected_company)
        return jsonify(data)

    return render_template('index.html', Companies=Company, Inches=Inches,ScreenResolution=ScreenResolution,Rams=Ram, Weight=Weight, Memory_Type=Memory_Type, Memory_Size=Memory_Size)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    products = request.form.get('products')
    TypeNames = request.form.get('TypeNames')
    Inches = request.form.get('Inches')
    ScreenResolution = request.form.get('ScreenResolution')
    Cpu = request.form.get('Cpu')
    Gpu = request.form.get('Gpu')
    Ram = request.form.get('Rams')
    OpSys = request.form.get('OpSys')
    Weight = request.form.get('Weight')
    Memory_Type = request.form.get('Memory_Type')
    Memory_Size = request.form.get('Memory_Size')

    SR = ScreenResolution.split()[-1].split("x")
    x_res = int(SR[0])
    y_res = int(SR[1])
    ScreenResolution = (x_res * y_res)

    PPi = (((x_res ** 2 + x_res ** 2) ** (1 / 2)) / float(Inches))

    try:
        prediction = model.predict(pd.DataFrame(columns=['Company', 'Product', 'TypeName', 'Inches', 'ScreenResolution',
                                                         'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight', 'Memory_Size',
                                                         'Memory_Type', 'PPi'],data=np.array([company, products, TypeNames, Inches, ScreenResolution,Cpu, Ram, Gpu, OpSys, Weight, Memory_Size,Memory_Type,PPi]).reshape(1, 13)))

        return str(np.round(prediction[0], 2))
    except:
        return str(" ( nothing found )")

if __name__ == "__main__":
    app.run(debug=True)
