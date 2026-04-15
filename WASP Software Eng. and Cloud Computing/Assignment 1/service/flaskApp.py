import openstack
from flask import Flask, jsonify, render_template

conn = openstack.connect(cloud='openstack')

CONTAINER  = 'YOUR-CONTAINER-NAME' 
OBJECT_NAME = 'counter.txt'

app = Flask(__name__)

@app.route('/')

def handle_get():

    # Read current number of visitors from your Swift container 
    try:
        response = conn.object_store.download_object(container=CONTAINER, obj=OBJECT_NAME)
        counter = int(bytes(response).decode('utf-8'))
    except:
        # Assume failure is because object doesn't exist yet 
        counter = 0
    
    counter += 1
    
    # Update
    conn.object_store.upload_object(container=CONTAINER, 
                                    name=OBJECT_NAME,
                                    data=str(counter),
                                    content_type='text/plain')

    return render_template('index.html', value=counter)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
