from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the core analysis function from your script.
# This will also initialize the models from emomain.py, which is what we want.
# This might take a moment when the server first starts.
try:
    # Import the necessary functions from your script
    from scripts.emomain import scrape_content, analyze_content
except Exception as e:
    print(f"AN ERROR OCCURRED DURING STARTUP: {e}")
    scrape_content, analyze_content = None, None

app = Flask(__name__)
CORS(app)  # This will allow the extension to call the server

@app.route('/analyze', methods=['POST'])
def analyze_url():
    # 1. Get the URL from the extension
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url']
    response_data = {}

    # 2. Run your Python logic if the imports were successful
    if scrape_content and analyze_content:
        print(f"---\
Received URL for analysis: {url}")
        
        # 2a. Scrape the content from the URL
        content_data = scrape_content(url)
        
        if not content_data or not content_data['content']:
            return jsonify({'error': f'Could not scrape content from {url}'}), 500

        # 2b. Analyze the scraped content
        emotion_data = analyze_content(content_data['content'])

        # 2c. Prepare the response
        if emotion_data:
            dominant_emotion = max(emotion_data, key=emotion_data.get)
            classification_status = f"Dominant emotion: {dominant_emotion}"
        else:
            emotion_data = {}
            classification_status = "Analysis could not be completed."

        response_data = {
            'emotions': emotion_data,
            'classification': {'status': classification_status}
        }
    else:
        # Fallback if the models couldn't be loaded
        response_data = {'error': 'Analysis models not available on the server.'}

    # 3. Return the final analysis to the extension
    return jsonify(response_data)

if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(port=5000, debug=True)

