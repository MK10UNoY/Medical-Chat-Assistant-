import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Suppress INFO and WARNING messages from TensorFlow
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle, pandas as pd
from flask import Flask, render_template, request

# Disease-to-number mapping
disease_mapping = {
    'Fungal infection': 15, 'Allergy': 4, 'GERD': 16, 'Chronic cholestasis': 9, 
    'Drug Reaction': 14, 'Peptic ulcer diseae': 33, 'AIDS': 1, 'Diabetes ': 12, 
    'Gastroenteritis': 17, 'Bronchial Asthma': 6, 'Hypertension ': 23, 
    'Migraine': 30, 'Cervical spondylosis': 7, 'Paralysis (brain hemorrhage)': 32, 
    'Jaundice': 28, 'Malaria': 29, 'Chicken pox': 8, 'Dengue': 11, 
    'Typhoid': 37, 'hepatitis A': 40, 'Hepatitis B': 19, 'Hepatitis C': 20, 
    'Hepatitis D': 21, 'Hepatitis E': 22, 'Alcoholic hepatitis': 3, 
    'Tuberculosis': 36, 'Common Cold': 10, 'Pneumonia': 34, 
    'Dimorphic hemmorhoids(piles)': 13, 'Heart attack': 18, 
    'Varicose veins': 39, 'Hypothyroidism': 26, 'Hyperthyroidism': 24, 
    'Hypoglycemia': 25, 'Osteoarthristis': 31, 'Arthritis': 5, 
    '(vertigo) Paroymsal  Positional Vertigo': 0, 'Acne': 2, 
    'Urinary tract infection': 38, 'Psoriasis': 35, 'Impetigo': 27
}

encodedSymp = {
    'nan': 126 ,'itching': 125, ' skin_rash': 95, ' continuous_sneezing': 23, ' shivering': 91, ' stomach_pain': 101,
    ' acidity': 2, ' vomiting': 115, ' indigestion': 47, ' muscle_wasting': 65, ' patches_in_throat': 77,
    ' fatigue': 39, ' weight_loss': 120, ' sunken_eyes': 102, ' cough': 24, ' headache': 42, ' chest_pain': 17,
    ' back_pain': 6, ' weakness_in_limbs': 117, ' chills': 18, ' joint_pain': 53, ' yellowish_skin': 124,
    ' constipation': 21, ' pain_during_bowel_movements': 72, ' breathlessness': 13, ' cramps': 25, ' weight_gain': 119,
    ' mood_swings': 61, ' neck_pain': 68, ' muscle_weakness': 66, ' stiff_neck': 100, ' pus_filled_pimples': 82,
    ' burning_micturition': 16, ' bladder_discomfort': 9, ' high_fever': 43, ' nodal_skin_eruptions': 69, ' ulcers_on_tongue': 112,
    ' loss_of_appetite': 57, ' restlessness': 88, ' dehydration': 27, ' dizziness': 32, ' weakness_of_one_body_side': 118,
    ' lethargy': 56, ' nausea': 67, ' abdominal_pain': 0, ' pain_in_anal_region': 73, ' sweating': 103, ' bruising': 15,
    ' cold_hands_and_feets': 19, ' anxiety': 5, ' knee_pain': 54, ' swelling_joints': 105, ' blackheads': 8, ' foul_smell_of urine': 41,
    ' skin_peeling': 94, ' blister': 10, ' dischromic _patches': 30, ' watering_from_eyes': 116, ' extra_marital_contacts': 36,
    ' diarrhoea': 29, ' loss_of_balance': 58, ' blurred_and_distorted_vision': 12, ' altered_sensorium': 4, ' dark_urine': 26,
    ' swelling_of_stomach': 106, ' bloody_stool': 11, ' obesity': 70, ' hip_joint_pain': 44, ' movement_stiffness': 62,
    ' spinning_movements': 98, ' scurring': 90, ' continuous_feel_of_urine': 22, ' silver_like_dusting': 92, ' red_sore_around_nose': 85,
    ' spotting_ urination': 99, ' passage_of_gases': 76, ' irregular_sugar_level': 50, ' family_history': 37,
    ' lack_of_concentration': 55, ' excessive_hunger': 35, ' yellowing_of_eyes': 123, ' distention_of_abdomen': 31,
    ' irritation_in_anus': 52, ' swollen_legs': 109, ' painful_walking': 74, ' small_dents_in_nails': 97, ' yellow_crust_ooze': 121,
    ' internal_itching': 49, ' mucoid_sputum': 63, ' history_of_alcohol_consumption': 45, ' swollen_blood_vessels': 107, ' unsteadiness': 113,
    ' inflammatory_nails': 48, ' depression': 28, ' fluid_overload': 40, ' swelled_lymph_nodes': 104, ' malaise': 59, ' prominent_veins_on_calf': 80,
    ' puffy_face_and_eyes': 81, ' fast_heart_rate': 38, ' irritability': 51, ' muscle_pain': 64, ' mild_fever': 60, ' yellow_urine': 122,
    ' phlegm': 78, ' enlarged_thyroid': 34, ' increased_appetite': 46, ' visual_disturbances': 114, ' brittle_nails': 14, ' drying_and_tingling_lips': 33,
    ' polyuria': 79, ' pain_behind_the_eyes': 71, ' toxic_look_(typhos)': 111, ' throat_irritation': 110, ' swollen_extremeties': 108,
    ' slurred_speech': 96, ' red_spots_over_body': 86, ' belly_pain': 7, ' receiving_blood_transfusion': 83, ' acute_liver_failure': 3,
    ' redness_of_eyes': 87, ' rusty_sputum': 89, ' abnormal_menstruation': 1, ' receiving_unsterile_injections': 84, ' coma': 20,
    ' sinus_pressure': 93, ' palpitations': 75}

app = Flask(__name__)

# Adding the MinMaxScaler
minmax_scaler = MinMaxScaler()
training_data=pd.read_csv('./X.csv')
minmax_scaler.fit(training_data)

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

def input_cleaner(form_inputs):
    """
    Cleans the 11 form inputs and arranges them in a 1D array.
    Missing values are replaced with np.nan and non-empty values are pushed to the beginning.
    """
    # Extract and clean inputs
    cleaned_inputs = [
        value.strip() if value.strip() else np.nan
        for key, value in form_inputs.items()
    ]

    # Filter out empty values (np.nan)
    non_empty_inputs = [value for value in cleaned_inputs if value is not np.nan]

    # Fill the rest of the array with np.nan to ensure the output length is 11
    final_array = non_empty_inputs + [np.nan] * (11 - len(non_empty_inputs))

    return final_array[:11]  # Ensure the array is exactly 11 elements long

def preprocess_inputs(inputs, vector_length=11):
    """Preprocess form inputs by modifying spaces in symptom names."""
    # Clean the input values (strip leading/trailing spaces)
    symptom_names = [value.strip() if isinstance(value, str) else None for value in inputs.values()]
    
    processed_symptoms = []
    for symptom in symptom_names:
        if symptom:
            # Remove space if the symptom is 'itching'
            if symptom == 'itching':
                processed_symptoms.append('itching')
            else:
                # Add a space in front of the symptom if it contains a space in the original input
                processed_symptoms.append(' ' + symptom if ' ' not in symptom else symptom)
        else:
            processed_symptoms.append('nan')
    
    return processed_symptoms

def predict_disease(symptom_names):
    """Model Predictor Blackbox"""
    # Encode symptoms using encodedSymp
    new_symptoms_encoded = [encodedSymp[symptom] for symptom in symptom_names if symptom in encodedSymp]

    # Reshape and normalize using minmax_scaler
    new_symptoms_encoded = np.array(new_symptoms_encoded).reshape(1, -1)
    print(new_symptoms_encoded)
    # errror part ################
    new_symptoms_normalized = minmax_scaler.transform(new_symptoms_encoded)
    print(new_symptoms_normalized)
    ##############################

    # Predict disease using the model
    prediction_probabilities = model.predict(new_symptoms_normalized)
    predicted_class = np.argmax(prediction_probabilities)

    reverse_disease_mapping = {v: k for k, v in disease_mapping.items()}

    # Get the disease name from reverse_disease_mapping
    predicted_disease = reverse_disease_mapping.get(predicted_class, "Unknown Disease")

    return predicted_disease

@app.route("/", methods=["GET", "POST"])
def predictor():
    # Initialize default variables
    inputs = {f"input{i}": "" for i in range(1, 12)}  # Default all inputs to empty strings
    prediction = None

    if request.method == "POST":
        # Fetch inputs from the form
        inputs = {f"input{i}": request.form.get(f"input{i}") for i in range(1, 12)}
        print(inputs)

        # Step 1: Clean the inputs using input_cleaner
        cleaned_inputs = input_cleaner(inputs)
        print(cleaned_inputs)

        # Step 2: Preprocess the cleaned inputs for the model
        preprocessed_inputs = preprocess_inputs(
            {f"input{i+1}": cleaned_inputs[i] for i in range(len(cleaned_inputs))}, 
            encodedSymp
        )
        print(preprocessed_inputs)

        # Ensure at least 3 non-empty symptoms are provided
        if sum(1 for value in preprocessed_inputs if value != 0) < 3:
            prediction = "Error: At least 3 symptoms are required to make a prediction."
        else:
            try:
                # Predict the disease using the predict_disease function
                predicted_disease = predict_disease(preprocessed_inputs)
                print("checkpoint")
                print(predict_disease)
                prediction = f"Possible Case of {predicted_disease}"
            except Exception as e:
                prediction = f"Error during prediction: {str(e)}"

    # Render the template with the prediction result and user inputs
    return render_template("index.html", prediction=prediction, inputs=inputs)

if __name__=="__main__":
    app.run(debug=True,port= 5002)