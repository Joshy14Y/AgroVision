import io

from PIL import Image


def test_predict_freshness(client, mock_freshness_service):
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    response = client.post(
        "/freshness/predict", files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "fresh_apple"
    assert data["confidence"] == 0.99
    mock_freshness_service.predict.assert_called_once()


def test_predict_invalid_image(client):
    response = client.post(
        "/freshness/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]
