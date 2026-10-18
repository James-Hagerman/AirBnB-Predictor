from flask import request
import tf_keras as keras
import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


MODEL = keras.models.load_model('AirBnB_Predictor/Keras_model', compile=False)

with open('AirBnB_Predictor/tokenizer.json') as f:
    TOKENIZER = keras.preprocessing.text.tokenizer_from_json(f.read())

MODEL([np.zeros((1, 10000)), np.zeros((1, 10000)), np.zeros((1, 170)), np.zeros((1, 14))], training=False)


def bow(text):
    """Function to convert description and amenities features into bow"""
    return TOKENIZER.texts_to_matrix(text)


def seq_embed(text):
    """Function to convert description and amenities features into embeddings"""
    seq = TOKENIZER.texts_to_sequences(text)
    max_seq_length = 170
    seq = keras.preprocessing.sequence.pad_sequences(
    seq, maxlen = max_seq_length, padding = 'post')
    return seq


def encode_desc_amenities(feature_dict):
    """Takes in feature_dict and returns bow representation and embeddings for the "description" and "amenities" features"""
    #Bag of words representation of descsription feature
    desc_bow = bow(feature_dict['description'])

    #Bag of words representation of amenities feature
    amenities_bow = bow(feature_dict['amenities'])

    #Embedding of description
    desc_embed = seq_embed(feature_dict['description'])

    #Embedding of amenities
    amenities_embed = seq_embed(feature_dict['amenities'])

    return desc_bow, amenities_bow, desc_embed, amenities_embed


def encode_cat_features(feature_dict):
    """
    Takes in dictionary of features and returns encoded categorical feature dataframe
    """
    #Dictionary of mappings to encode each categorical value
    replace_mapping = {"property_type": {"Apartment":0, "Bed & Breakfast":1, "Bungalow":2, "Condominium":3, "Guest House":4, "Hostel":5, "House":6, "Loft":7, "Other":8, "Townhouse":9}, 
                          "room_type": {"Entire House/apt":0, "Private room":1, "Shared room":2}, 
                          "bed_type": {"Airbed":0, "Couch":1, "Futon":2, "Pull-out sofa":3, "Real bed":4}, 
                          "cancellation_policy": {"Flexible":0, "Moderate":1, "Strict":2, "Super strict - 30":3, "Super strict - 60":4}, 
                           "city": {"Boston":0, "Chicago":1, "Washington DC":2, "Los Angeles":3, "New York City":4, "San Francisco":5}, 
                           }

    #Create dataframe of categorical features (all features except for 'description' and 'amenities')
    df = pd.DataFrame.from_dict(feature_dict)

    #Drop amenities and description columns
    df.drop(columns=['amenities', 'description'], inplace=True)

    #Encode all "Yes" and "No" values : Yes=1; No=0
    df.replace("Yes", 1, inplace=True)
    df.replace("No", 0, inplace=True)

    #Replace values according to replace_mapping dictionary
    df.replace(replace_mapping, inplace=True)

    df = df.astype(float)
    training_ranges = {"accommodates": (1, 16), "bathrooms": (0, 8), "bedrooms": (0, 10),
                       "beds": (0, 16), "number_of_reviews": (0, 542)}
    for column, (low, high) in training_ranges.items():
        df[column] = df[column].clip(low, high)

    return df


def predict_rate():
    """
    Grabs form data for each feature that the model takes and then
    appends all the feature data for that particular listing into a list called "feature_list", which is then fed into the model.

    Example feature_dict: 'property_type': ['House'], 'room_type': ['Shared room'], 'amenities': ['Cable TV, Wireless internet, free parking'], 
                            'accommodates': ['5'], 'bathrooms': ['3.0'], 'bed_type': ['Real bed'], 'cancellation_policy': ['Flexible'], 
                            'cleaning_fee': ['Yes'], 'city': ['Los Angeles'], 'description': ['Gentle breeze, awesome views'], 
                            'host_has_profile_pic': ['No'], 'host_identity_verified': ['Yes'], 'instant_bookable': ['Yes'], 
                            'number_of_reviews': ['50'], 'bedrooms': ['4'], 'beds': ['3']
    Example output: 213
    """
    model = MODEL

    features = ['property_type', 'room_type', 'amenities', 'accommodates', 'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'city', 
                'description', 'host_has_profile_pic', 'host_identity_verified', 
                'instant_bookable', 'number_of_reviews', 'bedrooms', 'beds']

    #Create dicionary of features, where the key is equal to feature name and value is the listing data
    feature_dict = {}
    for f in features:
        #If the value is a string int or float (ie '3'), cast it to int
        if request.values[f].isdigit():
            feature_dict[f] = [int(request.values[f])]
        else:
            feature_dict[f] = [request.values[f]]
    
    desc_bow, amenities_bow, desc_embed, amenities_embed = encode_desc_amenities(feature_dict)

    df = encode_cat_features(feature_dict)

    
    #Predict
    prediction = model([desc_bow, amenities_bow, desc_embed, df.astype(float).to_numpy()], training=False).numpy()

    return int(np.exp(prediction[0][0]))


if __name__ == "__main__":
    print(model)
