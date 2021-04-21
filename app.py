from flask import Flask, render_template, request, redirect, url_for
import pickle
from pythainlp.tokenize import word_tokenize
import pymysql

app = Flask(__name__)

conn = pymysql.connect(    
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '',
    db = 'data_db'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis', methods=['POST'])
def predict():
    classifier = pickle.load(open('sentiment.pkl', 'rb'))
    vocabulary = pickle.load(open('vocabulary.pkl', 'rb'))

    if request.method == 'POST':
        message = request.form['message']
        result = request.form['result']
        if message == '':
            return render_template('index.html', valid=0)
        else:
            data = [message]
            test_sentence = str(data)
            featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}
            print("test_sent:",test_sentence)
            res = classifier.classify(featurized_test_sentence)
            print("tag:",res)

            if res == 'pos':
                my_prediction = 1
            elif res == 'neg':
                my_prediction = 0
            else: my_prediction = 2
            
            if my_prediction == int(result):
                tg = 1
                print(tg)
            elif my_prediction != int(result):
                tg = 0
                print(tg)

            with conn.cursor() as cursors:
                ## เก็บค่าไว้ในตัวแปร ในที่นี้ เป็น string ทั้งหมด
                sql = "SELECT data_msg FROM data_tbl WHERE data_msg ='"+str(message)+"';"
                cursors.execute(sql)
                row = cursors.fetchall()

                if len(row) == 0:
                    sql2 = "Insert into `data_tbl` (`data_id`,`data_msg`,`data_con`) values(null,%s,%s)"
                    cursors.execute(sql2,(message,result))
                    conn.commit()

            return render_template('result.html', prediction=my_prediction, target=tg)

if __name__ == '__main__':
    app.run(debug=True)