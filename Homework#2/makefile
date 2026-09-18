all: bayesian density

.venv:
	python -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install numpy matplotlib scipy

bayesian: .venv
	.venv\Scripts\python "Bayssian_classifier.py"

density: .venv
	.venv\Scripts\python "DensityEstimation.py"

clean:
	rmdir /s /q .venv