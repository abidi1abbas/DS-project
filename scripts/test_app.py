"""Automated smoke test for the Streamlit website."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30)

app.run()
assert not app.exception, app.exception
assert len(app.text_area) == 1
assert len(app.button) == 1

app.text_area[0].set_value("yeh movie bohat achi hai")
app.button[0].click()
app.run()

assert not app.exception, app.exception
assert len(app.success) == 1
assert "Positive" in app.success[0].value

print("PASS: Streamlit app loads and returns a Positive prediction for the test input.")
