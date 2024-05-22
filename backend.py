from flask import Flask, request, jsonify,send_from_directory
#from some_spam_detection_module import check_if_spam  # 假设你有一个检查垃圾邮件的模块

app = Flask(__name__)

@app.route('/check-spam', methods=['POST'])
def check_spam():
    email = request.json.get('email')
    is_spam = True#check_if_spam(email)  # 你的垃圾邮件检测逻辑
    return jsonify({'isSpam': is_spam})
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
