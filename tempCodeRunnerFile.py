from flask import Flask, render_template ,url_for,request,session, redirect
app = Flask(__name__)

@app.route('/index' ,methods=['GET', 'POST'] )
def index():
  if 'username' in session:
    search_text = request.form['search_text']
    return redirect('/login')
  else:
    return render_template('/index.html')     

