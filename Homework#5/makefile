.venv:
	python -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install numpy matplotlib scipy pandas scikit-learn

problem1: .venv
	.venv\Scripts\python "problem1.py"

problem1b: .venv
	.venv\Scripts\python "problem1b.py"
	
problem2: .venv
	.venv\Scripts\python "problem2.py"
	


clean:
	rmdir /s /q .venv