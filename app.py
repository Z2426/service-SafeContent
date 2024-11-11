from flask import Flask
from controllers.content_controller import content_bp

app = Flask(__name__)

# Đăng ký blueprint cho content
app.register_blueprint(content_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
