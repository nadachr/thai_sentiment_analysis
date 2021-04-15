from flask import Flask, render_template, request, redirect, url_for
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import pymysql

app = Flask(__name__)

# pos.txt
with codecs.open('pos.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listpos=[e.strip() for e in lines]
del lines
f.close() # ปิดไฟล์

# neg.txt
with codecs.open('neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines]
f.close() # ปิดไฟล์

pos1=['pos']*len(listpos)
neg1=['neg']*len(listneg)

training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1))

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data]

classifier = nbc.train(feature_set)


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

# @app.route('/database')
# def showData():
#     with conn.cursor() as cur:
#         cur.execute("SELECT * FROM test_tbl")
#         row = cur.fetchall()
#         return render_template('show.html', data=row)

# @app.route("/insert",methods=['POST'])
# def insert():
#     if request.method=="POST":
#         message = request.form['message']
#         result = request.form['result']
    
#     with conn.cursor() as cursors:
#         ## เก็บค่าไว้ในตัวแปร ในที่นี้ เป็น string ทั้งหมด
#         sql="Insert into `test_tbl` (`test_id`,`test_del`,`test_con`) values(null,%s,%s)"
#         cursors.execute(sql,(message,result))
#         ## commit เป็นคำสั่งเปลี่ยนแปลงข้อมูลภายในฐานข้อมูล
#         conn.commit()
#         ## ให้เปลี่ยนแปลงหน้า หลังจากที่บันทึกเสร็จ โดยใช้ คำสั่ง url_for ให้เปลี่ยนไปที่ Route Showdata
#     return redirect(url_for('showData'))

if __name__ == '__main__':
    app.run(debug=True)