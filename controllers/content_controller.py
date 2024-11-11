from flask import Blueprint, request, jsonify
from models.content_model import check_sensitive_image, check_sensitive_text

content_bp = Blueprint('content', __name__)

@content_bp.route('/check-image', methods=['POST'])
def check_image():
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'image_url is required'}), 400

    is_sensitive, label = check_sensitive_image(image_url)
    if is_sensitive:
        return jsonify({'sensitive': True, 'label': label}), 200
    else:
        return jsonify({'sensitive': False}), 200

@content_bp.route('/check-text', methods=['POST'])
def check_text():
    data = request.json
    text_input = data.get('text')

    if not text_input:
        return jsonify({'error': 'text is required'}), 400

    is_sensitive = check_sensitive_text(text_input)
    if is_sensitive:
        return jsonify({'sensitive': True}), 200
    else:
        return jsonify({'sensitive': False}), 200
