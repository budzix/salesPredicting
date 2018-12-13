import pandas as pd
import re
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sklearn.feature_extraction.text import HashingVectorizer


DATA_DIR = "data/"
TEXT_VECTOR_SIZE = 128
BATCH_SIZE = 128
TARGET_MONTH = '2015-10'

sales = pd.read_csv(DATA_DIR + 'sales_train_v2.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
items = pd.read_csv(DATA_DIR + 'items.csv')
items.item_name = items.item_name.apply(lambda str: re.sub(r'[?|$|.|!|*|(|)|,|+|/|[|\]]', r'', str).lower())
items = items.drop(labels=['item_category_id'], axis=1)

vectorizer = HashingVectorizer(n_features=TEXT_VECTOR_SIZE)
temp = pd.DataFrame(data=vectorizer.fit_transform(items.item_name).todense())
temp['item_id'] = items['item_id']
items = pd.merge(items, temp).drop(labels=['item_name'], axis=1)
del temp

test = pd.read_csv(DATA_DIR + 'test.csv')
data_frame = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
data_frame = data_frame.drop(labels=['date_block_num', 'item_price'], axis=1)  # todo: we can add price later
data_frame = data_frame.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()

data_frame_test = pd.merge(test, data_frame, on=['item_id', 'shop_id'], how='left')
data_frame_test = data_frame_test.fillna(0)
text_data = pd.merge(data_frame_test, items, on='item_id', how='left').drop(labels=list(data_frame_test.columns.values), axis=1)
data_frame_test = data_frame_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)

labels = data_frame_test[TARGET_MONTH]
train_data = data_frame_test.drop(labels=[TARGET_MONTH], axis=1)

train_data = train_data.values
train_data = train_data.reshape((214200, 33, 1))

labels = labels.values
labels = labels.reshape(214200, 1)

text_data = text_data.values
text_data = text_data.reshape(214200, TEXT_VECTOR_SIZE)

main_input = Input(shape=(33, 1), dtype='float32', name='main_input')
lstm_out = LSTM(64, input_shape=(33, 1), dropout=0.0, recurrent_dropout=0.0)(main_input)
auxiliary_output = Dense(1, activation='linear', name='aux_output')(lstm_out)
auxiliary_input = Input(shape=(TEXT_VECTOR_SIZE,), name='text_vector')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
main_output = Dense(1, activation='linear', name='main_output')(x)
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error', 'accuracy'])

model.fit([train_data, text_data], [labels, labels],
          epochs=5)

test_data = data_frame_test.drop(labels=['2013-01'], axis=1)
test_data = test_data.values
test_data = test_data.reshape((214200, 33, 1))
predictions = model.predict([test_data, text_data])

predictions[1].clip(0., 20.)

predictions = pd.DataFrame(predictions[1], columns=['item_cnt_month'])
predictions.to_csv('submission.csv', index_label='ID')
