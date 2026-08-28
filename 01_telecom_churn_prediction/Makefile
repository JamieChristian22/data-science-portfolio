setup:
	pip install -r requirements.txt

data:
	python src/generate_data.py

train:
	python run_pipeline.py

test:
	pytest -q

app:
	streamlit run deployment/app_streamlit.py
