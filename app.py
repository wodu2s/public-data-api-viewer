from flask import Flask, request, jsonify, render_template
import requests
import json

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fetch", methods=["POST"])
def fetch_data():
    body = request.get_json()
    api_key = body.get("api_key", "").strip()
    endpoint = body.get("endpoint", "").strip()
    param_name = body.get("param_name", "serviceKey").strip()
    extra_params = body.get("extra_params", {})

    if not api_key:
        return jsonify({"success": False, "error": "API 키를 입력해주세요."}), 400
    if not endpoint:
        return jsonify({"success": False, "error": "API 엔드포인트 URL을 입력해주세요."}), 400

    params = {param_name: api_key, "_type": "json"}
    params.update(extra_params)

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        content_type = response.headers.get("Content-Type", "")

        # JSON 응답 시도
        if "json" in content_type:
            data = response.json()
        else:
            # XML이나 다른 포맷도 텍스트로 반환
            try:
                data = response.json()
            except Exception:
                data = response.text

        return jsonify({
            "success": True,
            "status_code": response.status_code,
            "data": data,
            "url": response.url,
        })

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "요청 시간이 초과되었습니다 (10초)."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "연결할 수 없습니다. URL을 확인해주세요."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
