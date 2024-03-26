from flask import Flask, render_template, request

import your_chatbot_script  # Import your chatbot code
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = your_chatbot_script.process_input(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)
