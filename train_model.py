import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
layers = keras.layers

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

train = train[:20000]
test = test[:15000]
# test = train[60000:74000]

def clean(df):
    """ 
    Cleans the dataset
    """

    df = df.copy()

    # encode binary columns
    df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1, 'f': 0})

    df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})

    df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})
    


    # Remove some of the unique property types
    df.loc[df['property_type'].isin(['Boutique hotel', 'Villa', 'Timeshare',
                                                    'In-law', 'Boat', 'Camper/RV', 'Castle', 'Cabin', 'Treehouse',
                                                    'Serviced apartment', 'Tipi', 'Hut', 'Vacation home', 'Tent',
                                                    'Chalet', 'Yurt', 'Earth House', 'Train', 'Cave', 'Guest suite', 'Dorm',
                                                    'Casa particular', 'Parking Space', 'Lighthouse', 'Island']), 'property_type'] = 'Other'
    
    # Creates price column that isnt in log form
#     e = 2.718281828
#     df['price'] = e ** df['log_price']
    
    # Drops useless columns 
    df = df.drop(['zipcode', 'thumbnail_url', 'host_response_rate',
                 'first_review', 'last_review', 'review_scores_rating', 'neighbourhood', 'id',
                 'longitude', 'latitude', 'host_since', 'name'], axis = 1)
    
    return df

# Applying clean function to both trian and test data
train = clean(train)
test = clean(test)

# Creating price column and dropping log price column from training data
e = 2.718281828
train['price'] = e ** train['log_price']

train = train.drop(['log_price'], axis = 1)

# Convert all categorical features in the training set to categories
train['property_type'] = train['property_type'].astype('category')
train['room_type'] = train['room_type'].astype('category')
train['bed_type'] = train['bed_type'].astype('category')
train['cancellation_policy'] = train['cancellation_policy'].astype('category')
train['cleaning_fee'] = train['cleaning_fee'].astype('category')
train['city'] = train['city'].astype('category')
# Converts all categorical features in the test set to categories
test['property_type'] = test['property_type'].astype('category')
test['room_type'] = test['room_type'].astype('category')
test['bed_type'] = test['bed_type'].astype('category')
test['cancellation_policy'] = test['cancellation_policy'].astype('category')
test['cleaning_fee'] = test['cleaning_fee'].astype('category')
test['city'] = test['city'].astype('category')

# Encodes all of the categorical features in the training set
cat_columns = train.select_dtypes(['category']).columns
cat_columns

train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)

train.head(500)

# Encodes all of the categorical features in the test set
cat_columns = test.select_dtypes(['category']).columns
cat_columns

test[cat_columns] = test[cat_columns].apply(lambda x: x.cat.codes)

train = train.dropna()
test = test.dropna()

# Splitting training data into our feature matrix and target vector
target = train['price']

# New Dataset for just our numeric data
X = train.drop(['price', 'amenities', 'description'], axis = 1)
y = np.log(target)

X_test = test.drop(['amenities', 'description'], axis = 1)

# Turns X and y into all floats for NN model
X = X.astype(float)
y = y.astype(float)
X_test = X_test.astype(float)

# Pulls out the description and amenities column for NLP processing 
desc_train = train['description'].astype(str)
desc_test = test['description'].astype(str)
amen_train = train['amenities'].astype(str)
amen_test = test['amenities'].astype(str)

#Creates tokenizer for NlP model for the Amenities and Description features
# 
vocab_size = 10000
tokenize = keras.preprocessing.text.Tokenizer(num_words=10000, char_level = False)
tokenize.fit_on_texts(desc_train)
tokenize.fit_on_texts(desc_test)         
tokenize.fit_on_texts(amen_train)
tokenize.fit_on_texts(amen_test)
with open('AirBnB_Predictor/tokenizer.json', 'w') as f:
    f.write(tokenize.to_json())

# Creates Bag of Words 
desc_bow_train = tokenize.texts_to_matrix(desc_train)
desc_bow_test = tokenize.texts_to_matrix(desc_test)
amen_bow_train = tokenize.texts_to_matrix(amen_train)
amen_bow_test = tokenize.texts_to_matrix(amen_test)

# Turns the tokenized description text into sequences 
train_desc_embed = tokenize.texts_to_sequences(desc_train)
test_desc_embed = tokenize.texts_to_sequences(desc_test)

max_seq_length = 170

# Train preprocessing for description
train_desc_embed = keras.preprocessing.sequence.pad_sequences(
train_desc_embed, maxlen = max_seq_length, padding = 'post')

# Test preprocessing for description
test_desc_embed = keras.preprocessing.sequence.pad_sequences(
test_desc_embed, maxlen = max_seq_length, padding = 'post')

# Turns the tokenized amenities text into sequences
train_amen_embed = tokenize.texts_to_sequences(amen_train)
test_amen_embed = tokenize.texts_to_sequences(amen_test)

max_seq_length = 170

train_amen_embed = keras.preprocessing.sequence.pad_sequences(
    train_amen_embed, maxlen = max_seq_length, padding = 'post')

test_amen_embed = keras.preprocessing.sequence.pad_sequences(
    test_amen_embed, maxlen = max_seq_length, padding = 'post')

# Multiple Input Layers to be concatenated
feat_set_1 = layers.Input(shape = (vocab_size,))
feat_set_2 = layers.Input(shape = (vocab_size,))
# numeric_set = layers.Input(shape = (14,))

merged_layer1 = layers.Concatenate(axis = 1)([feat_set_1,
                                  feat_set_2])
#                                   numeric_set])
# Output layer
# variety_inputs = layers.Input(shape = (14,))

merged_layer = layers.Dense(256, activation = 'relu')(merged_layer1)

predictions = layers.Dense(1, activation = 'linear')(merged_layer)

wide_model = keras.Model(inputs = [feat_set_1,
                                   feat_set_2],
#                                    numeric_set],
                        outputs = predictions)

wide_model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['MeanAbsoluteError'])

# Deep model
deep_inputs = layers.Input(shape = (max_seq_length,), name = 'desc_seq')
numeric_set = layers.Input(shape = (14,), name = 'listing_features')

normalizer = keras.layers.Normalization()
normalizer.adapt(X.values)
numeric_scaled = normalizer(numeric_set)

embedding = layers.Embedding(vocab_size, 8, input_length = max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)

deep_merged = layers.concatenate([embedding, numeric_scaled])
deep_hidden = layers.Dense(64, activation = 'relu')(deep_merged)
embed_out = layers.Dense(1)(deep_hidden)
deep_model = keras.Model(inputs = [deep_inputs, numeric_set] , outputs = embed_out)
print(deep_model.summary())

deep_model.compile(loss = 'mse',
                  optimizer = 'adam',
                  metrics = ['MeanSquaredError'])

# Combine wide and deep model into one model
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(inputs = [feat_set_1, feat_set_2, deep_inputs, numeric_set], outputs = merged_out)

print(combined_model.summary())

combined_model.compile(loss='mse',
                       optimizer='adam',
                       metrics=['MeanAbsoluteError'])

combined_model.fit([desc_bow_train, amen_bow_train, train_desc_embed, X],
                    y,
                    epochs = 40,
                   batch_size = 128)

combined_model.save('AirBnB_Predictor/Keras_model')

reconstructed_model = keras.models.load_model('AirBnB_Predictor/Keras_model')

check_inputs = [desc_bow_test[:5], amen_bow_test[:5], test_desc_embed[:5], X_test[:5]]
before = combined_model.predict(check_inputs).ravel()
after = reconstructed_model.predict(check_inputs).ravel()
print("Predicted nightly prices for 5 test listings:", np.round(np.exp(after), 2))
assert np.allclose(before, after, rtol=1e-4), "Reloaded model does not match the trained model"
print("Saved model and tokenizer.json are ready in AirBnB_Predictor/")
