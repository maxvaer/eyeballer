import os
import pytest

from eyeballer.model import EyeballModel
from eyeballer.visualization import HeatMap


@pytest.fixture
def model():
    weights_file = "tests/models/test_weights.h5"
    if not os.path.isfile(weights_file):
        pytest.skip("Symlink the latest weights file to " + weights_file)

    model_kwargs = {
        "weights_file": weights_file,
        "print_summary": False,
        "seed": None,
        "quiet": True
    }
    return EyeballModel(**model_kwargs)


def test_different_seed_predict(model):
    model_kwargs = {
        "weights_file": None,
        "print_summary": False,
        "seed": 12345678,
        "quiet": True
    }
    same_seed_model = EyeballModel(**model_kwargs)

    screenshot = "tests/data/404.png"

    results_one = model.predict(screenshot)
    results_two = same_seed_model.predict(screenshot)

    assert results_one != results_two


def test_same_seed_predict():
    model_kwargs = {
        "weights_file": None,
        "print_summary": False,
        "seed": 12345678,
        "quiet": True
    }
    same_seed_model = EyeballModel(**model_kwargs)

    screenshot = "tests/data/404.png"

    results_one = same_seed_model.predict(screenshot)
    results_two = same_seed_model.predict(screenshot)

    assert results_one == results_two


def test_predict_custom404(model):
    screenshot = "tests/data/404.png"
    results = model.predict(screenshot)[0]
    assert results["custom404"] > 0.5


def test_predict_not_custom404(model):
    screenshot = "tests/data/nothing.png"
    results = model.predict(screenshot)[0]
    assert results["custom404"] < 0.5


def test_predict_login(model):
    screenshot = "tests/data/login.png"
    results = model.predict(screenshot)[0]
    assert results["login"] > 0.5


def test_predict_not_login(model):
    screenshot = "tests/data/nothing.png"
    results = model.predict(screenshot)[0]
    assert results["login"] < 0.5


def test_predict_homepage(model):
    screenshot = "tests/data/homepage.png"
    results = model.predict(screenshot)[0]
    assert results["webapp"] > 0.5


def test_predict_not_homepage(model):
    screenshot = "tests/data/nothing.png"
    results = model.predict(screenshot)[0]
    assert results["webapp"] < 0.5


def test_predict_oldlooking(model):
    screenshot = "tests/data/old-looking.png"
    results = model.predict(screenshot)[0]
    assert results["oldlooking"] > 0.5


def test_predict_not_oldlooking(model):
    screenshot = "tests/data/nothing.png"
    results = model.predict(screenshot)[0]
    assert results["oldlooking"] < 0.5


def test_file_doesnt_exist(model):
    screenshot = "tests/data/doesnotexist.png"
    with pytest.raises(FileNotFoundError):
        model.predict(screenshot)


def test_folder(model):
    screenshots = "tests/data/"
    results = model.predict(screenshots)
    assert len(results) == 7


def test_file_is_empty(model):
    """
    We're just testing that it doesn't crash, basically
    """
    screenshot = "tests/data/empty.png"
    model.predict(screenshot)


def test_file_is_invalid(model):
    """
    We're just testing that it doesn't crash, basically
    """
    screenshot = "tests/data/invalid.png"
    model.predict(screenshot)


def test_heatmap(model):
    screenshot = "tests/data/login.png"
    HeatMap(screenshot, model, 0.5)

    screenshot = "tests/data/404.png"
    HeatMap(screenshot, model, 0.5)

    screenshot = "tests/data/empty.png"
    HeatMap(screenshot, model, 0.5)
