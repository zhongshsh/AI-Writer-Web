import os
from flask import Flask, render_template, request, jsonify
from infer import Writer


model = Writer()
base_dir = os.environ.get("BASE_DIR", "")
app = Flask(__name__, static_url_path=base_dir + "/static")


@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", base_dir=base_dir)
    else:
        data = request.form
        outmsg = model.inference(data)
        return jsonify(outmsg)


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html"), 500


@app.errorhandler(404)
def server_error(error):
    return render_template("notFound.html", base_dir=base_dir), 404


if __name__ == "__main__":
    app.run("0.0.0.0")
