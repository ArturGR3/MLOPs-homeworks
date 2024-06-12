# Import the Flask module and jsonify function from the flask package
from flask import Flask, jsonify

# Create an instance of the Flask class for our web app
app = Flask(__name__)

# Define a route decorator to tell Flask what URL should trigger our function
@app.route('/ping', methods=['GET'])
def ping():
    # The function to be run when the '/ping' route is hit by a GET request
    # It returns a JSON response with a message "pong"
    return jsonify({"message": "pong"}), 200

# A Python conditional statement that checks if this module is the main program
if __name__ == '__main__':
    # Run the app on a local development server
    # debug=True provides more error details if something goes wrong
    app.run(debug=True)